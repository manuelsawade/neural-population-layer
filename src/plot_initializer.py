import numpy as np
import matplotlib.pyplot as plt

# Number of "neurons"
n_neurons = 100
x = np.arange(1, n_neurons + 1)

# Generate data
normal_dist = np.random.normal(0, 1, n_neurons)
uniform_dist = np.random.uniform(-1, 1, n_neurons)
sine_layer = np.sin(np.linspace(0, 2 * np.pi * 8, n_neurons))

# Create figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

# Plot normal distribution
axes[0].plot(x, normal_dist, 'o', color='tab:blue', linewidth=0)
axes[0].set_title("Random Normal")
axes[0].set_xlabel("Hidden Layer Neurons")
axes[0].set_ylabel("Preferred Values")
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot uniform distribution
axes[1].plot(x, uniform_dist, 'o', color='tab:orange', linewidth=0)
axes[1].set_title("Random Uniform")
axes[1].set_xlabel("Hidden Layer Neurons")
axes[1].grid(True, linestyle=':', alpha=0.6)

# Plot sine layer
axes[2].plot(x, sine_layer, color='tab:green')
axes[2].set_title("Sine Layer")
axes[2].set_xlabel("Hidden Layer Neurons")
axes[2].grid(True, linestyle=':', alpha=0.6)

# Adjust layout and save
plt.tight_layout()
plt.savefig("preferred_value_initializations.png", dpi=300)
plt.close()
