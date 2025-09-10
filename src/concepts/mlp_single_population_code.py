import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random

from datasets.mnist import MNIST

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

N_POP_OUT = 30   
SIGMA_OUT = 1.5
BATCH_SIZE = 128
HIDDEN_DIM = 64     
N_POP_PER_FEATURE = 5 
SIGMA_HIDDEN = 0.5   

preferred_out = torch.linspace(0, 9, N_POP_OUT)

def target_population_code(labels, N=N_POP_OUT, sigma=SIGMA_OUT):
    labels_exp = labels.float().unsqueeze(1)
    mus = preferred_out.unsqueeze(0)
    return torch.exp(-0.5 * ((labels_exp - mus) / sigma) ** 2)

class HiddenPopulationEncode(nn.Module):
    """
    Expand a vector of scalar features into population-coded representation.
    Each scalar feature is encoded with N_pop neurons with preferred values in [-1,1].
    """
    def __init__(self, num_features, n_pop_per_feature=N_POP_PER_FEATURE, sigma=SIGMA_HIDDEN):
        super().__init__()
        self.num_features = num_features
        self.n_pop = n_pop_per_feature
        self.sigma = sigma

        prefs = torch.linspace(-1.0, 1.0, n_pop_per_feature)
        self.register_buffer("prefs", prefs)  # (n_pop,)
        
    def forward(self, x):
        B = x.size(0)

        x_norm = torch.tanh(x) 
        x_exp = x_norm.unsqueeze(2) 
        prefs = self.prefs.view(1, 1, self.n_pop) 
        pop = torch.exp(-0.5 * ((x_exp - prefs) / self.sigma) ** 2)  
        return pop.view(B, -1) 

class PopHiddenMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=HIDDEN_DIM, out_pop=N_POP_OUT):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_pop = HiddenPopulationEncode(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim * N_POP_PER_FEATURE, 128)
        self.fc3 = nn.Linear(128, out_pop)  # output population code
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden = F.relu(self.fc1(x))       
        hidden_pop = self.hidden_pop(hidden)  
        h2 = F.relu(self.fc2(hidden_pop))
        out = F.relu(self.fc3(h2))  
        return out, hidden_pop 

transform = transforms.Compose([transforms.ToTensor()])
mnist = MNIST()
train_dataset, test_dataset = mnist(training_noise=0.0) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PopHiddenMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(50):  
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        out_pop, hidden_pop = model(data)
        tgt_pop = target_population_code(target).to(device)
        loss = loss_fn(out_pop, tgt_pop)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()
all_hidden_pops = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        out_pop, hidden_pop = model(data)
        all_hidden_pops.append(hidden_pop.cpu())
        all_targets.append(target)
all_hidden_pops = torch.cat(all_hidden_pops, dim=0)
all_targets = torch.cat(all_targets, dim=0)

decoder = nn.Linear(all_hidden_pops.size(1), N_POP_OUT)
dec_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
dec_loss_fn = nn.MSELoss()

dataset_hidden = torch.utils.data.TensorDataset(all_hidden_pops, target_population_code(all_targets))
loader_hidden = DataLoader(dataset_hidden, batch_size=64, shuffle=True)

for epoch in range(50):
    dec_loss = 0.0
    for hpop, tgt in loader_hidden:
        hpop, tgt = hpop.to(device), tgt.to(device)
        dec_opt.zero_grad()
        pred = decoder(hpop)
        loss = dec_loss_fn(pred, tgt)
        loss.backward()
        dec_opt.step()
        dec_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Decoder epoch {epoch+1}, Loss: {dec_loss/len(loader_hidden):.4f}")

with torch.no_grad():
    preds_pop = decoder(all_hidden_pops.to(device)).cpu()
    decoded_vals = (preds_pop * preferred_out.unsqueeze(0)).sum(dim=1) / (preds_pop.sum(dim=1) + 1e-8)
    decoded_round = decoded_vals.round().clamp(0,9).long()

accuracy = (decoded_round == all_targets).float().mean().item()
print(f"Decoder accuracy from hidden population (rounded): {accuracy*100:.2f}%")

n_show = 8
fig, axes = plt.subplots(n_show, 2, figsize=(6, 2*n_show))
test_iter = iter(test_loader)
imgs, labs = next(test_iter)
with torch.no_grad():
    out_pop_batch, hidden_pop_batch = model(imgs.to(device))
    pred_hidden_pop = decoder(hidden_pop_batch.cpu().to(device)).cpu()
    decoded_vals_batch = (pred_hidden_pop * preferred_out.unsqueeze(0)).sum(dim=1) / (pred_hidden_pop.sum(dim=1)+1e-8)
for i in range(n_show):
    axes[i,0].imshow(imgs[i].squeeze(), cmap="gray")
    axes[i,0].axis('off')
    axes[i,0].set_title(f"True: {int(labs[i].item())}")
    axes[i,1].text(0.1, 0.6, f"Decoded cont.: {decoded_vals_batch[i].item():.2f}\nDecoded rounded: {int(decoded_vals_batch[i].round().item())}", fontsize=12)
    axes[i,1].axis('off')

plt.tight_layout()
plt.show()

decoded_values = decoded_vals[:20].numpy()
true_values = all_targets[:20].numpy()
print("First 20 true vs decoded (continuous):")
for t, d in zip(true_values, decoded_values):
    print(f"{t} -> {d:.2f}")
