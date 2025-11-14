import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

orientations = (-2, 2)  # orientation range for the neuron population

# Define the x-range and parameters
x = np.linspace(orientations[0], orientations[1], 12)
x_smooth = np.linspace(orientations[0], orientations[1], 200)
input_position_1 = -2  # Original input value
input_position_2 = 0
input_position_3 = 2 # Perturbed input value
sigma = 1.0
cmap = cm.get_cmap("plasma")

# Gaussian activation function
def gaussian_activation(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Compute the activations for both input positions
activation_1 = gaussian_activation(x, input_position_1, sigma)
activation_2 = gaussian_activation(x, input_position_2, sigma)
activation_3 = gaussian_activation(x, input_position_3, sigma)

activation_1 /= activation_1.max()
activation_2 /= activation_2.max()
activation_3 /= activation_3.max()

activation_1_smooth = gaussian_activation(x_smooth, input_position_1, sigma)
activation_2_smooth = gaussian_activation(x_smooth, input_position_2, sigma)
activation_3_smooth = gaussian_activation(x_smooth, input_position_3, sigma)

# Normalize both activation curves
activation_1_smooth /= activation_1_smooth.max()
activation_2_smooth /= activation_2_smooth.max()
activation_3_smooth /= activation_3_smooth.max()

# Weighted average decoder
def weighted_average_decoder(activation_1, x):
    # Calculate the weighted sum (weights are the activations themselves)
    #weights_1 = activation_1 / np.sum(activation_1)
    #weights_2 = activation_2 / np.sum(activation_2)
    #decoded_value = np.sum(weights_1 * x) ##+ np.sum(weights_2 * x)

    # denom = np.sum(activation_1) + 1e-8
    # cont = np.sum(activation_1 * x) / denom
    # decoded_value = np.clip(cont, -4, 4)

    denom = np.sum(activation_1, axis=-1) + 1e-8
    cont = np.sum(activation_1 * x, axis=-1) / denom
    decoded_value = np.clip(cont, orientations[0], orientations[1])

    if decoded_value == -0.0:
        decoded_value = 0.0

    # denom = out.sum(dim=-1) + 1e-8
    # cont = (out * mu).sum(dim=-1, keepdim=False) / denom
    # rounded = cont.clamp(orientation[0], orientation[1])

    #return rounded

    return decoded_value

# Decoding the value
decoded_value1 = weighted_average_decoder(activation_1, x)
decoded_value2 = weighted_average_decoder(activation_2, x)
decoded_value3 = weighted_average_decoder(activation_3, x)

# Create the figure with 3 plots
fig, axes = plt.subplots(2, 3, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 0.4]})

# --- Plot 1: Original input and its tuning curve ---
axes[0, 0].plot(x_smooth, activation_1_smooth, color='gray', linewidth=2)
axes[0, 0].axvline(input_position_1, color='gray', linestyle='--')
axes[0, 0].plot(input_position_1, 1.0, 'o', color='black', markersize=6)
axes[0, 0].set_title(f"Original Input = {input_position_1}")
axes[0, 0].set_xticks([])
axes[0, 0].set_xlim(orientations[0], orientations[1])
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].set_ylabel("Activation")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle=':', alpha=0.6)

# --- Plot 1 (Heatmap) ---
axes[1, 0].imshow(np.expand_dims(activation_1_smooth, axis=0), extent=[x_smooth.min(), x_smooth.max(), 0, 1], cmap=cmap, aspect='auto', origin='lower')
axes[1, 0].set_yticks([])
axes[1, 0].set_xlabel(f"Activity = {decoded_value1:.2f}")
axes[1, 0].set_xlim(orientations[0], orientations[1])
axes[1, 0].grid(False)

# --- Plot 2: Perturbed input and its tuning curve ---
axes[0, 1].plot(x_smooth, activation_2_smooth, color='black', linewidth=2)
axes[0, 1].axvline(input_position_2, color='black', linestyle='--')
axes[0, 1].plot(input_position_2, 1.0, 'o', color='black', markersize=6)
axes[0, 1].set_title(f"Perturbed Input = {input_position_2}")
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 1].set_xlim(orientations[0], orientations[1])
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)

# --- Plot 2 (Heatmap) ---
axes[1, 1].imshow(np.expand_dims(activation_2_smooth, axis=0), extent=[x_smooth.min(), x_smooth.max(), 0, 1], cmap=cmap, aspect='auto', origin='lower')
axes[1, 1].set_yticks([])
axes[1, 1].set_xlabel(f"Activity = {decoded_value2:.2f}")
axes[1, 1].set_xlim(orientations[0], orientations[1])
axes[1, 1].grid(False)

# --- Plot 3: Weighted Average Decoder ---
axes[0, 2].plot(x_smooth, activation_3_smooth, color='gray', linewidth=2)
axes[0, 2].axvline(input_position_3, color='black', linestyle='--')
axes[0, 2].plot(input_position_3, 1.0, 'o', color='black', markersize=6)
axes[0, 2].set_title(f"Perturbed Input = {input_position_3}")
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])
axes[0, 2].set_xlim(orientations[0], orientations[1])
axes[0, 2].set_ylim(0, 1.1)
axes[0, 2].grid(True, linestyle=':', alpha=0.6)

# --- Plot 2 (Heatmap) ---
axes[1, 2].imshow(np.expand_dims(activation_3_smooth, axis=0), extent=[x_smooth.min(), x_smooth.max(), 0, 1], cmap=cmap, aspect='auto', origin='lower')
axes[1, 2].set_yticks([])
axes[1, 2].set_xlabel(f"Activity = {decoded_value3:.2f}")
axes[1, 2].set_xlim(orientations[0], orientations[1])
axes[1, 2].grid(False)

# Adjust layout and save
plt.tight_layout(h_pad=0.3, w_pad=0.4)
plt.savefig("tuning_curves_with_decoder.png", dpi=300)
plt.close(fig)

print("Figure saved as tuning_curves_with_decoder.png")
