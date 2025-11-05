import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Optional reproducibility
np.random.seed(42)

# =============================
# 1️⃣  Preferred vs Input with Inverted Distance Activation
# =============================

n_neurons = 100
x = np.arange(0.5, n_neurons + 0.5)
x_smooth = np.linspace(0.5, n_neurons + 0.5, 300)
# Input values (new random)
input_values = np.random.uniform(-1, 1, n_neurons)
cmap = cm.get_cmap("plasma")
# Compute softmax to find active region
softmax = np.exp(input_values) / np.sum(np.exp(input_values))
max_idx = np.argmax(softmax)
max_x = x[max_idx]

# Gaussian tuning curve centered at the most active neuron
sigma = 10.0 # adjust width of tuning curve
gaussian = np.exp(-0.5 * ((x - max_x) / sigma)**2)
# Normalize tuning curve to input range (0 to max input)
gaussian = gaussian / np.max(gaussian) * np.max(input_values)

# Plot 2
fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 0.4]})
ax2 = axes[0]
ax2.plot(x, input_values, 'x', color='gray', label='Input Value')
ax2.plot(x, gaussian, '-', color='gray', linewidth=2)
ax2.axvline(max_x, color='gray', linestyle=':', label='Peak Activity')

ax2.set_ylabel("Activation / Value")
ax2.set_title("Softmax Peak and Tuning Curve Activation")
ax2.set_xlim(0, n_neurons)
ax2.set_xticks([])
ax2.legend(['Input Value', 'Tuning Curve', 'Peak Activity'], loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

ax_heat = axes[1]
ax_heatmap = np.expand_dims(gaussian, axis=0)
ax_heat.imshow(ax_heatmap, extent=[x.min() -0.5, x.max() +0.5, 0, 1],
                cmap=cmap, aspect='auto', origin='lower')
ax_heat.set_yticks([])
ax_heat.set_xlabel(f"Hidden Layer Neurons = {n_neurons}")
ax_heat.set_xlim(0, n_neurons)
#ax_heat.set_xlim(orientations[i][0], orientations[i][1])
ax_heat.grid(False)

plt.tight_layout()
plt.savefig("softmax_gaussian_tuning.png", dpi=300)
plt.close(fig)

print("Saved: softmax_gaussian_tuning.png")
