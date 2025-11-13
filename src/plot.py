import numpy as np
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from datasets.mnist import MNIST
from scipy.interpolate import make_interp_spline

def sine_base(freq, phase, amp, x_pos, x_size):
    return amp * (torch.sin(2 * torch.pi * freq * x_pos / x_size + phase) + 1)


def plot_mnist_sample():
    training_data, _ = MNIST()(training_noise=0.8)
    image, label = training_data[0]

    plt.imshow(image.squeeze().numpy(), cmap='gray')
    #plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

def plot_smooth_tensor(ax, tensor, color="blue", title=""):
    tensor_size = tensor.size(dim=-1)
    # print(tensor_size)
    # print(torch.arange(tensor_size).float().numpy())
    # print(tensor.numpy())

    neurons = torch.arange(tensor_size).float().numpy()
    
    spline = make_interp_spline(neurons, tensor.numpy())

    X = np.linspace(neurons.min(), neurons.max(), 16)
    Y = spline(X)

    ax.plot(X, Y, color=color)
    ax.set_title(title)

def plot_oscillatory_population():
    freq = 4.0
    phase = 0.5

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, hspace=0.5)
    axs = gs.subplots(sharey=True)
    axs[0].set_title('Input')
    
    x = torch.randn(1, 16)
    #plot_smooth_tensor(axs[0], x.squeeze())  # Example input
    
    x_size = x.size(dim=1)
    #x_norm = x / x.max(dim=-1, keepdim=True).values

    plot_smooth_tensor(axs[0], x.squeeze(), title="x")

    positions = torch.arange(x_size).float()

    theta = 2 * torch.pi * freq * positions / x_size + phase
    sine_mask = 1 * torch.sin(theta)

    sine_mask = sine_mask / x.max(dim=-1, keepdim=True).values.squeeze()

    #plot_smooth_tensor(axs[2], sine_mask)
    #sine_mask = sine_mask / sine_mask.max(dim=-1, keepdim=True).values

    plot_smooth_tensor(axs[1], sine_mask)

    plot_smooth_tensor(axs[2], (sine_mask + 1))
    # plt.plot(sine_mask)
    # plt.show()
    
    masked = sine_mask + x
    #masked = masked / masked.max(dim=-1, keepdim=True).values
    # print(f'   x: {x}')
    # print(f'mask: {sine_mask}')
    # print(f' out: {masked}')
    plot_smooth_tensor(axs[3], masked.squeeze())
    plt.show()

def plot_dynamic_population():
    neurons = 100
    
    torch.manual_seed(1997)
    alpha = 1.0
    sigma = 0.5

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(3, hspace=0.5)
    axs = gs.subplots()
    axs[0].set_title('Input')
    
    x = torch.randn(1, neurons)
    print("x", x)
    plot_smooth_tensor(axs[0], x.squeeze(), title="x")  # Example input
    
    p = F.softmax(alpha * x, dim=1)
    #p = (p + 1e-8) / p.max(dim=-1, keepdim=True).values
    print("p", p)
    plot_smooth_tensor(axs[1], p.squeeze(), title="softmax") 

    positions = torch.arange(x.size(dim=1)).unsqueeze(0)
    print("pos", positions)
    #mu = torch.sum(p * positions, keepdim=True, dim=-2)
    mu = torch.sum(p * positions, dim=1, keepdim=True)
    #mu = 2 * mu - 1
    #plot_smooth_tensor(axs[2], mu, title="mu") 
    print("mu", mu)

    #mask = (1 - ((x - mu) / sigma) ** 2) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    mask = torch.exp(-0.5 * ((x - p) / sigma) ** 2)
    #plot_smooth_tensor(axs[2], mask.squeeze(), title="dist")
    print("mask", mask) 

    mask_norm = (mask + 1e-8) / mask.max(dim=-1, keepdim=True).values
    #mask_norm = 2 * mask_norm - 1
    print("dist_norm", mask_norm)
    plot_smooth_tensor(axs[2], mask_norm.squeeze(), color="purple") 
    plot_smooth_tensor(axs[2], x.squeeze(), color="blue", title="dist_norm") 

    #a_norm = x / x.max(dim=-1, keepdim=True).values  
    #plot_smooth_tensor(axs[4], a_norm.squeeze(), title="x_norm")   

    masked_a = (x + (mask_norm - x))
    #masked_a = mask_norm * a_norm
    # print("out", masked_a)
    # plot_smooth_tensor(axs[3], masked_a.squeeze(), color="red") 
    # plot_smooth_tensor(axs[3], x.squeeze(), title="x + (dist - x_norm)") 
    #plt.tight_layout()
    plt.savefig("dynamic_population.png")

def plot_fixed_population():
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, hspace=0.5)
    axs = gs.subplots()
    
    freq = 8.0
    phase = 0.5
    amp = 1
    sigma = -1.0
    eps = 1e-8

    x = torch.randn(1, 16)
    plot_smooth_tensor(axs[0], x[0].squeeze())
    
    # x = torch.tensor([
    #     list(range(16)), 
    #     list(range(16, 32))])
    
    print(x)
    # print(torch.sum(x))
    # print(torch.sum(x, dim=-1))
    # print(torch.sum(x, dim=-2))
    # print(torch.sum(x ** 2, dim=-2))

    x_size = x.size(dim=1)
    x_pos = torch.arange(x_size, device=x.device)
    #plot_smooth_tensor(axs[0], x.squeeze())
    
    #x = x / x.max(dim=-1, keepdim=True).values

    mask = sine(freq, phase, amp, x_pos, x_size)
    print(mask)
    plot_smooth_tensor(axs[1], mask)
    #mask_norm = mask / mask.max(dim=-1, keepdim=True).values
    
    diff = x - mask
    print(diff)
    numerator = torch.sum(diff ** 2, dim=-2)
    print(numerator)
    denom = torch.sum(x ** 2, dim=-1) + torch.sum(mask ** 2, dim=-1)
    print(denom)
    denom = denom.unsqueeze(1)
    print(denom)
    dists2 = numerator / (denom + eps)
    print(dists2)

    print(f'd: {dists2}')
    plot_smooth_tensor(axs[2], dists2[0])

    dists2 = dists2 / dists2.max(dim=-1, keepdim=True).values
    print(f'd: {dists2}')
    plot_smooth_tensor(axs[3], dists2[0])

    #plt.show()

def sine_base(freq, phase, amp, x_pos, x_size):
        return amp * (torch.sin(2 * torch.pi * freq * x_pos / x_size + phase) + 1)

def sine(freq, phase, amp, x_pos, x_size): 
        return amp * torch.sin(2 * torch.pi * freq * x_pos / x_size + phase)

#0.5*((norm((x-mean(x))-(y-mean(y)))^2)/(norm(x-mean(x))^2+norm(y-mean(y))^2))

#plot_mnist_sample()
#plot_oscillatory_population()
#plot_fixed_population()
plot_dynamic_population()