import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from datasets.mnist import MNIST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Population Coding Layer with multi-dim trainable μ ====
class PopulationCodeLayer(nn.Module):
    def __init__(self, in_features, num_neurons, sigma=0.8, init_min=0.0, init_max=1.0):
        super().__init__()
        self.fc_to_stim = nn.Linear(in_features, 2)
        self.mu = nn.Parameter(torch.empty(num_neurons, 2))
        nn.init.uniform_(self.mu, init_min, init_max)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

    def forward(self, x):
        stim = self.fc_to_stim(x)  # [batch, stim_dim]
        diff = stim.unsqueeze(1) - self.mu.unsqueeze(0)  # [batch, num_neurons, stim_dim]
        dist_sq = (diff ** 2).sum(dim=-1)  # [batch, num_neurons]
        out = torch.exp(-0.5 * dist_sq / (self.sigma ** 2))
        #out = F.softmax(out)
        return out
    
class PopulationActivation(nn.Module):
    def __init__(self, num_features, neurons_per_feature=3, sigma=0.5):
        super().__init__()
        self.num_features = num_features
        self.neurons_per_feature = neurons_per_feature
        
        self.mu = nn.Parameter(torch.randn(num_features, neurons_per_feature))
        self.log_sigma = nn.Parameter(torch.log(torch.ones(num_features, neurons_per_feature) * sigma), requires_grad=True) 
    
    def forward(self, x):
        x_expanded = x.unsqueeze(-1) 
        mu = self.mu.unsqueeze(0) 
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        
        responses = torch.exp(-0.5 * ((x_expanded - mu) / sigma)**2)
        
        return responses.view(x.size(0), self.num_features * self.neurons_per_feature)

# ==== Small model with two population layers ====
class PopNet(nn.Module):
    def __init__(self, size=100,size2=50, neurons_per_feature=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, size)
        self.pop1 = PopulationActivation(size, neurons_per_feature=neurons_per_feature)
        self.fc2 = nn.Linear(size * neurons_per_feature, size2)
        self.pop2 = PopulationActivation(size2, neurons_per_feature=neurons_per_feature)
        self.classifier = nn.Linear(size2 * neurons_per_feature, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.pop1(x)
        x = self.fc2(x)
        x = self.pop2(x)
        x = self.classifier(x)
        return x

BATCH_SIZE = 32
transform = transforms.Compose([transforms.ToTensor()])

mnist = MNIST()
train_dataset, test_dataset = mnist(training_noise=1.0) 

#train_dataset = Subset(train_dataset, list(range(10000)))
#test_dataset = Subset(test_dataset, list(range(1000)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Training ====
model = PopNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(40):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # test accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: test acc = {acc*100:.2f}% loss = {loss.item():.4f}")
