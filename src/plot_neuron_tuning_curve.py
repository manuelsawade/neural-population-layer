import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Settings
orientations = [(-4,4), (-2,2), (-3,3)]  # three neurons with same 
input_positions = [-2, 0, 2]  # three input stimuli
sigmas = [1.0, 1.0, 1.2]
neurons = [12, 6, 8]
cmap = cm.get_cmap("magma")
enum = iter("abcdefg")
# Create figure with 3 columns × 2 rows (top = curve, bottom = heatmap)
fig, axes = plt.subplots(3, 3, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 0.4, 0.4]})

for i, pos in enumerate(input_positions):
    x = np.linspace(orientations[i][0], orientations[i][1], 200)
    # Compute Gaussian activation
    activation = np.exp(-0.5 * ((x - pos) / sigmas[i]) ** 2)
    activation /= activation.max()  # normalize to [0,1]
    
    # --- Top: tuning curve ---
    ax_curve = axes[0, i]
    ax_curve.plot(x, activation, color='gray', linewidth=2)
    ax_curve.axvline(pos, color='black', linestyle='--')
    ax_curve.plot(pos, 1.0, 'o', color='black', markersize=5)
    ax_curve.set_xlim(orientations[i][0], orientations[i][1])
    ax_curve.set_ylim(0, 1.1)
    ax_curve.set_title(f"Input = {pos}")
    ax_curve.grid(True, linestyle=':', alpha=0.6)
    ax_curve.set_xticks([])

    ax_curve.text(
    0.02,        # a little left of the axes
    0.92,               # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=ax_curve.transAxes
)

    ax_curve.legend([rf'$\sigma = {sigmas[i]}$'], loc='lower right')

    if i == 0:
        ax_curve.set_ylabel("Activation")

    # --- Bottom: heatmap ---
    ax_heat = axes[1, i]
    ax_heatmap = np.expand_dims(activation, axis=0)
    ax_heat.imshow(ax_heatmap, extent=[x.min(), x.max(), 0, 1],
                   cmap=cmap, aspect='auto', origin='lower')
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    #ax_heat.set_xlabel("Output Units")
    ax_heat.set_xlim(orientations[i][0], orientations[i][1])
    ax_heat.grid(False)

    out = np.linspace(orientations[i][0], orientations[i][1], neurons[i]) 

    activation_out = np.exp(-0.5 * ((out - pos) / sigmas[i]) ** 2)
    activation_out /= activation_out.max() 


    ax_out = axes[2, i]
    ax_out_heatmap = np.expand_dims(activation_out, axis=0)
    ax_out.imshow(ax_out_heatmap, extent=[out.min(), out.max(), 0, 1],
                   cmap=cmap, aspect='auto', origin='lower')
    ax_out.set_yticks([])
    ax_out.set_xlabel(f"Population Neurons = {neurons[i]}")
    ax_out.set_xlim(orientations[i][0], orientations[i][1])
    ax_out.grid(False)

# Adjust layout and save
plt.tight_layout(h_pad=0.3, w_pad=0.4)
plt.savefig("neuron_tuning_curve.png", dpi=300)
plt.close(fig)

print("Figure saved as three_gaussian_tuning_with_heatmaps.png")
