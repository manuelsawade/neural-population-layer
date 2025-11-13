import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F

# Optional reproducibility
np.random.seed(1995)
torch.manual_seed(1997)

# =============================
# 1️⃣  Preferred vs Input with Inverted Distance Activation
# =============================

n_neurons = 100
alpha = 20.0

x = torch.linspace(0.5, n_neurons+ 0.5, steps=n_neurons).unsqueeze(0)
print("x", x)

input_values = torch.randn(1, n_neurons)
input_values = F.normalize(input_values, p=float("inf"))

cmap = cm.get_cmap("plasma")
# Compute softmax to find active region
softmax = F.softmax(alpha * input_values)

# Gaussian tuning curve centered at the most active neuron
sigma = 0.1 # adjust width of tuning curve
gaussian = (1 - ((input_values - softmax) / sigma) ** 2) * torch.exp(-0.5 * ((input_values - softmax) / sigma) ** 2)
#gaussian = torch.exp(-0.5 * ((input_values - softmax) / sigma)**2) #/ (2 * sigma ** 2)
print("gaussian", gaussian)
# Normalize tuning curve to input range (0 to max input)
#gaussian = gaussian / gaussian.max(dim=-1, keepdim=True).values# * np.max(input_values)

#softmax = (softmax - softmax.min()) / (softmax.max() - softmax.min())

# Plot 2
fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 0.4]}, sharex=True)
ax2 = axes[0]
ax2.plot(x.squeeze(), input_values.squeeze(), 'x', color='gray', label='Input Value')
ax2.plot(x.squeeze(), softmax.squeeze(), ':', color='purple', linewidth=3, label=rf'Softmax ($\alpha = {alpha}$)')
ax2.plot(x.squeeze(), gaussian.squeeze(), '-', color='black', linewidth=1, label=rf"Mexican Hat ($\sigma = {sigma}$)")
#ax2.axvline(max_x, color='gray', linestyle=':', label='Peak Activity')

ax2.set_ylabel("Activation / Value")
ax2.set_title("Softmax Peak and Tuning Curve Activation")
ax2.set_xlim(0, n_neurons)
#ax2.set_xticks([])
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.6)

ax_heat = axes[1]
ax_heatmap = np.expand_dims(gaussian.squeeze(), axis=0)
ax_heat.imshow(ax_heatmap, extent=[x.min() - 0.5, x.max() + 0.5, 0, 1],
                cmap=cmap, aspect='auto', origin='lower')
ax_heat.set_yticks([])
ax_heat.set_xlabel(f"Hidden Layer Neurons = {n_neurons}")
ax_heat.set_xlim(0, n_neurons)
#ax_heat.set_xlim(orientations[i][0], orientations[i][1])
ax_heat.grid(False)

plt.tight_layout()
plt.savefig("softmax_mexican_hat.png", dpi=300)
plt.close(fig)

print("Saved: softmax_gaussian_tuning.png")
