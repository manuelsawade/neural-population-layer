# import numpy as np
# import matplotlib.pyplot as plt

# # Number of hidden neurons
# n_neurons = 30
# x = np.arange(1, n_neurons + 1)

# # Preferred values: sine distribution
# preferred_values = np.sin(np.linspace(0, 2 * np.pi, n_neurons))

# # Example random input values (uniform)
# input_values = np.random.uniform(-1, 1, n_neurons)

# # Create figure
# plt.figure(figsize=(8, 4))

# # Plot both distributions
# plt.plot(x, preferred_values, 'o', color='tab:green', label='Preferred Value')
# plt.plot(x, input_values, 'o', color='tab:orange', label='Input Value')

# # Draw distance lines
# for i in range(n_neurons):
#     plt.plot([x[i], x[i]], [preferred_values[i], input_values[i]],
#              color='gray', linestyle='--', linewidth=0.8)

# # Labels and title
# plt.xlabel("Hidden Layer Neurons")
# plt.ylabel("Value")
# plt.title("Distance Between Preferred and Input Values")
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.6)

# # Save to disk
# plt.tight_layout()
# plt.savefig("sine_vs_random_distances.png", dpi=300)
# plt.close()

# print("Figure saved as sine_vs_random_distances.png")





# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors

# # Number of hidden neurons
# n_neurons = 30
# x = np.arange(1, n_neurons + 1)

# # Preferred values: sine distribution
# preferred_values = np.sin(np.linspace(0, 2 * np.pi, n_neurons))

# # Example random input values (uniform)
# input_values = np.random.uniform(-1, 1, n_neurons)

# # Compute distances
# distances = np.abs(preferred_values - input_values)

# # Normalize distances for colormap
# norm = mcolors.Normalize(vmin=np.min(distances), vmax=np.max(distances))
# cmap = plt.get_cmap("YlOrRd_r")

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(8, 4))

# # Plot both distributions
# ax.plot(x, preferred_values, 'o', color='black', label='Preferred (Sine)')
# ax.plot(x, input_values, 'x', color='gray', label='Input (Random)')

# # Draw color-coded distance lines
# for i in range(n_neurons):
#     color = cmap(norm(distances[i]))
#     ax.plot([x[i], x[i]], [preferred_values[i], input_values[i]],
#             color=color, linewidth=2)

# #a = distances / np.max(distances)

# # Add colorbar (attached to the same axis)
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax)
# cbar.ax.set_yticklabels(['1', '0'])
# cbar.set_label('Distance Magnitude')
# cbar.ax.invert_yaxis()

# # Labels and title
# ax.set_xlabel("Hidden Layer Neurons")
# ax.set_ylabel("Value")
# ax.set_title("Preferred vs. Input Values with Distance Heat Mapping")
# ax.legend()
# ax.grid(True, linestyle=':', alpha=0.6)

# # Save to disk
# plt.tight_layout()
# plt.savefig("sine_vs_random_distances_heat.png", dpi=300)
# plt.close(fig)

# print("Figure saved as sine_vs_random_distances_heat.png")






import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Optional: reproducibility
np.random.seed(42)

# Number of hidden neurons
n_neurons = 50
x = np.arange(0.5, n_neurons + 0.5)

# Preferred values: sine distribution
preferred_values = np.random.uniform(-1, 1, n_neurons)#np.sin(np.linspace(0, 2 * np.pi * 2, n_neurons))

# Example random input values (uniform)
input_values = np.random.uniform(-1, 1, n_neurons)

# Compute absolute distances
distances = np.abs(preferred_values - input_values)

# Normalize distances to [0,1]
dist_min = distances.min()
dist_max = distances.max()
if dist_max - dist_min == 0:
    norm_dist = np.zeros_like(distances)
else:
    norm_dist = (distances - dist_min) / (dist_max - dist_min)

# Invert normalized distances to get activation: short distance -> high activation
activations = 1.0 - norm_dist  # range [0,1], 1 = highest activation

# Choose a colormap where high activation -> "hot" visual (warm colors)
cmap = cm.get_cmap("plasma")   # plasma goes from dark->warm; good for hot mapping
act_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)  # activation normalization

# Create figure and axis
fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 0.4]})

# Plot both distributions
axes[0].plot(x, preferred_values, 'o', color='black', label='Preferred Value', markersize=6)
axes[0].plot(x, input_values, 'x', color='gray', label='Input Value', markersize=6)

# Draw color-coded distance lines using activation for color mapping
for i in range(n_neurons):
    color = cmap(act_norm(activations[i]))
    axes[0].plot([x[i], x[i]], [preferred_values[i], input_values[i]],
            color=color, linewidth=2)

# Add colorbar representing Activation (high = hot)
# sm = cm.ScalarMappable(cmap=cmap, norm=act_norm)
# sm.set_array([])  # required for colorbar
# cbar = fig.colorbar(sm, ax=axes[0], pad=0.02)
# cbar.set_label('Activation')

# Labels and title
#ax.set_xlabel("Hidden Layer Neurons")
axes[0].set_ylabel("Value")
axes[0].set_title("Activation With Random Uniform Initialization")
axes[0].legend(loc='upper right')
axes[0].set_xlim(0, n_neurons)
axes[0].set_xticks([])
axes[0].grid(True, linestyle=':', alpha=0.6)

ax_heat = axes[1]
ax_heatmap = np.expand_dims(activations, axis=0)
ax_heat.imshow(ax_heatmap, extent=[x.min() -0.5, x.max() + 0.5, 0, 1],
                cmap=cmap, aspect='auto', origin='lower')
ax_heat.set_yticks([])
ax_heat.set_xlabel("Hidden Layer Neurons = 50")
ax_heat.set_xlim(0, n_neurons)
#ax_heat.set_xlim(orientations[i][0], orientations[i][1])
ax_heat.grid(False)

# Save to disk
plt.tight_layout()
plt.savefig("distance_inverted_activation_uni.png", dpi=300)
plt.close(fig)

print("Figure saved as: sine_vs_random_activations_inverted.png")

