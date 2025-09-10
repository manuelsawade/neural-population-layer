import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt

from datasets.mnist import MNIST

SEED = 100
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

N_POP_OUT = 30
SIGMA_OUT = 1.5
BATCH_SIZE = 64
H1 = 50
H2 = 10
N_POP_PER_FEATURE = 5
SIGMA_HIDDEN = 0.8
LR = 0.001 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA1 = 0.6
LAMBDA2 = 0.6

EPOCHS_BASELINE = 20
EPOCHS_AUX = 20
PROBE_EPOCHS = 100
PROBE_LR = 0.005

preferred_out = torch.linspace(0,9,N_POP_OUT).to(DEVICE)

def target_population_code(labels):
    labels_exp = labels.float().unsqueeze(1)
    mus = preferred_out.unsqueeze(0)
    return torch.exp(-0.5 * ((labels_exp - mus) / SIGMA_OUT) ** 2)

class HiddenPopulationEncode(nn.Module):
    def __init__(self, num_features, n_pop_per_feature=N_POP_PER_FEATURE, sigma=SIGMA_HIDDEN):
        super().__init__()
        self.n_pop = n_pop_per_feature; self.sigma = sigma
        prefs = torch.linspace(-1.0, 1.0, n_pop_per_feature)
        self.register_buffer("prefs", prefs)

    def forward(self, x):
        #x_norm = torch.tanh(x)
        x_exp = x.unsqueeze(2)
        prefs = self.prefs.view(1,1,self.n_pop)
        pop = torch.exp(-0.5 * ((x_exp - prefs) / self.sigma) ** 2)
        return pop.view(x.size(0), -1)
    
class TrainablePopulationCodeLayer(nn.Module):
    def __init__(self, in_features, num_neurons, init_min=0.0, init_max=9.0, sigma=1.0, 
                 stim_dim=10):
        super().__init__()
        self.sigma = sigma
        self.mu = nn.Parameter(torch.empty(num_neurons, stim_dim))
        nn.init.uniform_(self.mu, init_min, init_max)
        self.fc = nn.Linear(in_features, stim_dim)  # map input features to scalar stimulus value

    def forward(self, x):
        stim = self.fc(x)
        diff = stim.unsqueeze(1) - self.mu.unsqueeze(0)  # [batch, num_neurons, stim_dim]
        dist_sq = (diff ** 2).sum(dim=-1)                # squared distance in stim space
        out = torch.exp(-0.5 * dist_sq / self.sigma**2) 
        return out
    
class TrainPopMLP(nn.Module): 
    def __init__(self, input_dim=28*28, h1=H1, h2=H2, output_dim=N_POP_OUT):
        super().__init__()
        self.tl1 = TrainablePopulationCodeLayer(input_dim, h1)
            
    def forward(self, x):
        x = x.view(x.size(0), -1)
        p1 = self.tl1(x)
        out = F.relu(p1)

        return out


class TwoPopMLP_withAux(nn.Module):
    def __init__(self, input_dim=28*28, h1=H1, h2=H2, output_dim=N_POP_OUT, attach_aux=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.pop1 = HiddenPopulationEncode(h1)
        self.fc2 = nn.Linear(h1 * N_POP_PER_FEATURE, h2)
        self.pop2 = HiddenPopulationEncode(h2)
        self.fc3 = nn.Linear(h2 * N_POP_PER_FEATURE, N_POP_OUT)
        self.fc_out = nn.Linear(output_dim, N_POP_OUT)
        self.attach_aux = attach_aux
        if attach_aux:
            self.aux1 = nn.Linear(h1 * N_POP_PER_FEATURE, N_POP_OUT)
            self.aux2 = nn.Linear(h2 * N_POP_PER_FEATURE, N_POP_OUT)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #h1 = F.relu(self.fc1(x))
        p1 = self.pop1(self.fc1(x))# self.pop1(self.fc1(x))
        #h2 = F.relu(self.fc2(p1))
        p2 = self.pop2(self.fc2(p1))
        #h3 = F.relu(self.fc3(p2))
        out = self.fc3(p2)#F.relu(self.fc3(p2))
        if self.attach_aux:
            aux1_out = F.relu(self.aux1(p1))
            aux2_out = F.relu(self.aux2(p2))
            return out, p1, p2, aux1_out, aux2_out
        else:
            return out, p1, p2

def train_linear_decoder(Xpop, Ypop, epochs=PROBE_EPOCHS, lr=PROBE_LR, device=DEVICE):
    ds = torch.utils.data.TensorDataset(Xpop, Ypop)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    dec = nn.Linear(Xpop.size(1), N_POP_OUT).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dec.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = dec(xb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
    return dec

def pop_decode_to_digit(pop_acts):
    denom = pop_acts.sum(dim=1, keepdim=True) + 1e-8
    cont = (pop_acts * preferred_out.unsqueeze(0)).sum(dim=1).squeeze() / denom.squeeze(1)
    rounded = cont.round().clamp(0,9).long()
    return cont, rounded

transform = transforms.Compose([transforms.ToTensor()])

mnist = MNIST()
train_dataset, test_dataset = mnist(training_noise=0.0) 

train_dataset = Subset(train_dataset, list(range(10000)))
test_dataset = Subset(test_dataset, list(range(1000)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loss_fn = nn.CrossEntropyLoss()

model_aux = TrainPopMLP().to(DEVICE)
opt_aux = torch.optim.Adam(model_aux.parameters(), lr=LR)
for ep in range(EPOCHS_AUX):
    model_aux.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model_aux(xb)
        loss = loss_fn(pred, yb)

        loss.backward()
        opt_aux.step()
        for param in model_aux.parameters():
            param.grad = None

        running += loss.item()
    print(f"Epoch {ep+1}/{EPOCHS_AUX} total loss {running/len(train_loader):.4f}")

model_aux.eval()
P1a_list=[]; P2a_list=[]; OUTa_list=[]; Ya_list=[]

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model_aux(xb)
        OUTa_list.append(out.cpu()); Ya_list.append(yb)


# dec_p1_a = train_linear_decoder(P1a, target_population_code(Ya).to('cpu'))
# dec_p2_a = train_linear_decoder(P2a, target_population_code(Ya).to('cpu'))

# with torch.no_grad():
#     cont_out_a, r_out_a = pop_decode_to_digit(OUTa)

# acc_out_a = (r_out_a == Ya).float().mean().item()
# print("Aux decoder accs (rounded): P1={:.2f}%, P2={:.2f}%, OUT={:.2f}%".format(acc_p1_a*100, acc_p2_a*100, acc_out_a*100))

# layers = ['Pop1','Pop2','Out']
# aux_accs = [acc_p1_a*100, acc_p2_a*100, acc_out_a*100]

# x = np.arange(len(layers)); width = 0.35
# fig, ax = plt.subplots(figsize=(7,4))
# ax.bar(x+width/2, aux_accs, width)
# ax.set_xticks(x); ax.set_xticklabels(layers)
# ax.set_ylabel("Decoder accuracy (%) (rounded)")
# ax.set_title("Decoder accuracy by layer")
# ax.legend(["Baseline","With Aux"])
# plt.tight_layout()
# plt.show()

# print("Aux accs:", aux_accs)
