import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions ---

def gaussian_tuning(x, mu, sigma=15, height=1.0):
    """Gaussian tuning curve."""
    return height * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def poisson_spikes(rate, T=1000, dt=1):
    """Simulate Poisson spike trains for given rate (Hz)."""
    n_steps = int(T / dt)
    p_spike = rate * dt / 1000.0  # rate is in Hz, T in ms
    return np.random.rand(n_steps) < p_spike

# --- Setup stimulus space ---
x = np.linspace(0, 180, 200)  # e.g. orientation in degrees

# 1. Classical neuroscience: population of neurons with tuning curves
neurons_pref = np.linspace(20, 160, 5)
pop_curves = [gaussian_tuning(x, mu, sigma=20) for mu in neurons_pref]

# 2. Dynamic Field Theory: one continuous activation bump (represents population)
dft_field = gaussian_tuning(x, 90, sigma=25, height=1.2)

# 3. Deep learning population code (output layer representation)
dl_outputs = [gaussian_tuning(x, mu, sigma=10, height=1.0) for mu in [60, 90, 120]]
dl_code = np.sum(dl_outputs, axis=0)  # combined population output

# 4. Spiking Neural Networks: tuning via spike trains
rates = [gaussian_tuning(90, mu, sigma=20, height=30) for mu in neurons_pref]
spike_trains = [poisson_spikes(rate, T=500) for rate in rates]
firing_rates = [np.mean(train) * 1000 for train in spike_trains]  # Hz

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Classical neuroscience
for curve, mu in zip(pop_curves, neurons_pref):
    axs[0, 0].plot(x, curve, label=f"Neuron {int(mu)}°")
axs[0, 0].set_title("a) Gaussian Tuning Curve")
axs[0, 0].set_xlabel("Stimulus (orientation, °)")
axs[0, 0].set_ylabel("Activity over time")
axs[0, 0].legend()

# Dynamic field theory
axs[0, 1].plot(x, dft_field, color="darkred", lw=2)
axs[0, 1].set_title("b) Field Activity")
axs[0, 1].set_xlabel("Feature space (orientation, °)")
axs[0, 1].set_ylabel("Activation")

# Deep learning population code
for out in dl_outputs:
    axs[1, 0].plot(x, out, linestyle="--", alpha=0.7)
axs[1, 0].plot(x, dl_code, color="blue", lw=2, label="Population code (sum)")
axs[1, 0].set_title("c) Output Population Code")
axs[1, 0].set_xlabel("Output dimension")
axs[1, 0].set_ylabel("Activation")
axs[1, 0].legend()

# Spiking neural networks
axs[1, 1].bar([f"{int(mu)}°" for mu in neurons_pref], firing_rates, color="gray")
axs[1, 1].set_title("d) Estimated Tuning Curve")
axs[1, 1].set_ylabel("Continuous Activity")

plt.tight_layout()
plt.savefig("population_code_domains.png")
plt.close()
