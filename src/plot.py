import numpy as np
import torch

from matplotlib import pyplot as plt
from data.mnist import MNIST
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

def plot_smooth_tensor(ax, tensor):
    tensor_size = tensor.size(dim=-1)
    # print(tensor_size)
    # print(torch.arange(tensor_size).float().numpy())
    # print(tensor.numpy())

    neurons = torch.arange(tensor_size).float().numpy()
    
    spline = make_interp_spline(neurons, tensor.numpy())

    X = np.linspace(neurons.min(), neurons.max(), 500)
    Y = spline(X)

    ax.plot(X, Y)

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

    plot_smooth_tensor(axs[0], x.squeeze())

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
plot_fixed_population()