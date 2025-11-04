import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Optional reproducibility
np.random.seed(42)

# =============================
# 1️⃣  Preferred vs Input with Inverted Distance Activation
# =============================

n_neurons = 30
x = np.arange(1, n_neurons + 1)

# Preferred values: sine distribution
# preferred_values = np.sin(np.linspace(0, 2 * np.pi, n_neurons))
# # Input values: random uniform
# input_values = np.random.uniform(-1, 1, n_neurons)

# # Compute distances and normalize
# distances = np.abs(preferred_values - input_values)
# dist_min, dist_max = distances.min(), distances.max()
# norm_dist = (distances - dist_min) / (dist_max - dist_min) if dist_max != dist_min else np.zeros_like(distances)

# # Activation = inverted distance
# activations = 1.0 - norm_dist  # short distance -> high activation

# # Colormap setup
# cmap = cm.get_cmap("plasma")
# act_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

# # Plot 1
# fig1, ax1 = plt.subplots(figsize=(10, 4))
# ax1.plot(x, preferred_values, 'o-', color='tab:green', label='Preferred (Sine)')
# ax1.plot(x, input_values, 'o-', color='tab:orange', label='Input (Random)')

# # Distance lines color-coded by activation
# for i in range(n_neurons):
#     color = cmap(act_norm(activations[i]))
#     ax1.plot([x[i], x[i]], [preferred_values[i], input_values[i]], color=color, linewidth=2)

# sm = cm.ScalarMappable(cmap=cmap, norm=act_norm)
# sm.set_array([])
# cbar = fig1.colorbar(sm, ax=ax1, pad=0.02)
# cbar.set_label('Activation (short distance → high activation)')

# ax1.set_xlabel("Hidden Layer Neurons")
# ax1.set_ylabel("Value")
# ax1.set_title("Preferred vs Input Values — Activation Inverted from Distance")
# ax1.legend(loc='upper right')
# ax1.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
# plt.savefig("sine_vs_random_activations_inverted.png", dpi=300)
# plt.close(fig1)

# print("Saved: sine_vs_random_activations_inverted.png")

# =============================
# 2️⃣  Softmax Peak & Gaussian Tuning Curve
# =============================

# Input values (new random)
input_values = np.random.uniform(-1, 1, n_neurons)

# Compute softmax to find active region
softmax = np.exp(input_values) / np.sum(np.exp(input_values))
max_idx = np.argmax(softmax)
max_x = x[max_idx]

# Gaussian tuning curve centered at the most active neuron
sigma = 2.5 # adjust width of tuning curve
gaussian = np.exp(-0.5 * ((x - max_x) / sigma)**2)
# Normalize tuning curve to input range (0 to max input)
gaussian = gaussian / np.max(gaussian) * np.max(input_values)

# Plot 2
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(x, input_values, 'o', color='gray', label='Input Value')
ax2.plot(x, softmax, 'o--', color='tab:blue', label='Softmax')
ax2.plot(x, gaussian, '-', color='tab:red', linewidth=2, label='Tuning Curve')
ax2.axvline(max_x, color='gray', linestyle=':', label='Peak Activity')

ax2.set_xlabel("Hidden Layer Neurons")
ax2.set_ylabel("Activation / Value")
ax2.set_title("Softmax Peak and Tuning Curve Activation")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("softmax_gaussian_tuning.png", dpi=300)
plt.close(fig2)

print("Saved: softmax_gaussian_tuning.png")
