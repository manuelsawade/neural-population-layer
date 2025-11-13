import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F

from display_names import get_display_name

np.random.seed(1995)
torch.manual_seed(1997)

def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma)**2)

def mexican_hat(x, mu, sigma):
    return (1 - ((x - mu) / sigma) ** 2) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

n_neurons = 50

alpha_values = [10.0, 10.0, 10.0, 5.0]
sigma_values = [0.2, 0.3, 0.2, 0.1]
activation = [gaussian, gaussian, mexican_hat, mexican_hat]
titles = ["gaussian", "side", "mexican_hat", "none"]
legend = ["gaussian", "gaussian", "mexican_hat", "mexican_hat"]

x = torch.linspace(0.5, n_neurons+ 0.5, steps=n_neurons).unsqueeze(0)
print("x", x)

input_values = torch.randn(1, n_neurons)
input_values = F.normalize(input_values, p=float("inf"))

cmap = cm.get_cmap("plasma")

fig, axes = plt.subplots(4, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 0.4, 3, 0.4]}, sharex=True)

fig.supxlabel(f'Hidden Layer Neurons = {n_neurons}', y=0.02)
fig.suptitle(f"Activation Based On Softmax Distribution", y=0.98)
fig.supylabel("Activation / Value")

ax_iter = [[axes[0][0], axes[1][0]], [axes[2][0], axes[3][0]], [axes[0][1], axes[1][1]], [axes[2][1], axes[3][1]]]
for ax, activation, alpha, sigma, title, legend in zip(ax_iter, activation, alpha_values, sigma_values, titles, legend):
    softmax = F.softmax(alpha * input_values)
    output = activation(input_values, softmax, sigma)

    ax2 = ax[0]
    ax2.plot(x.squeeze(), input_values.squeeze(), 'x', color='gray', label='Input Value')
    ax2.plot(x.squeeze(), softmax.squeeze(), ':', color='purple', linewidth=3, label=rf'Softmax ($\alpha = {alpha}$)')
    ax2.plot(x.squeeze(), output.squeeze(), '-', color='black', linewidth=1, label=rf"{get_display_name(legend)} ($\sigma = {sigma}$)")
    #ax2.axvline(max_x, color='gray', linestyle=':', label='Peak Activity')

    #ax2.set_ylabel("Activation / Value")
    if title in "gaussian" or title in "mexican_hat":
        ax[0].set_title(f"{get_display_name(title)} Activation")

    if title in "mexican_hat" or title in "none":
        ax[0].set_yticks([])

    ax[0].legend(loc='lower right')


    ax2.set_xlim(0, n_neurons)
    #ax2.set_xticks([])
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax_heat = ax[1]
    ax_heatmap = np.expand_dims(output.squeeze(), axis=0)
    ax_heat.imshow(ax_heatmap, extent=[x.min() - 0.5, x.max() + 0.5, 0, 1],
                    cmap=cmap, aspect='auto', origin='lower')
    ax_heat.set_yticks([])
    #ax_heat.set_xlabel(f"Hidden Layer Neurons = {n_neurons}")
    ax_heat.set_xlim(0, n_neurons)
    #ax_heat.set_xlim(orientations[i][0], orientations[i][1])
    ax_heat.grid(False)

#fig.legend(loc='lower right')
#ax2.set_ylabel("Activation / Value")

plt.tight_layout()
plt.savefig("softmax_gaussian_tuning.png", dpi=300)
plt.close(fig)

print("Saved: softmax_gaussian_tuning.png")
