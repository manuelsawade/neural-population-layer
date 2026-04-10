import numpy as np
import matplotlib.pyplot as plt

# ---------------- Synthetic Data Generation ----------------

np.random.seed(42)

T = 2.0    # seconds
fs = 1000  # sampling rate
t = np.linspace(0, T, int(T * fs))

# Two different activity patterns (populations A and B)
pattern_A = 3 + 1.5 * np.sin(2 * np.pi * 3 * t) + np.random.normal(0, 0.2, len(t))
pattern_B = 2 + 1.3 * np.sin(2 * np.pi * 5 * t + 1.0) + np.random.normal(0, 0.2, len(t))

# Spike generator with contributions from A and B
n_neurons = 40
spike_trains = []

for i in range(n_neurons):
    # Spike rate influenced by both patterns
    rate = 10 + 3 * (pattern_A / np.max(pattern_A)) + 3 * (pattern_B / np.max(pattern_B))
    prob = (rate / np.max(rate)) * 0.02  # convert to spike probability per ms

    spikes = t[np.random.rand(len(t)) < prob]
    spike_trains.append(spikes)

# Compute difference activity (A - B)
pattern_diff = pattern_A - pattern_B

# --------------------- Plotting ----------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Plot 1: Activity Pattern A
axes[0, 0].plot(t, pattern_A, color="tab:blue")
axes[0, 0].set_title("Activity Pattern A")
axes[0, 0].set_ylabel("Activity")
axes[0, 0].grid(True, linestyle=':')

# Plot 2: Activity Pattern B
axes[0, 1].plot(t, pattern_B, color="tab:red")
axes[0, 1].set_title("Activity Pattern B")
axes[0, 1].grid(True, linestyle=':')

# Plot 3: Spike Train (All activity)
for i, spikes in enumerate(spike_trains):
    axes[1, 0].vlines(spikes, i + 0.5, i + 1.5, color="black")

axes[1, 0].set_ylim(0.5, n_neurons + 0.5)
axes[1, 0].set_title("Spike Train (All Activity)")
axes[1, 0].set_ylabel("Neuron")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].grid(True, linestyle=':')

# Plot 4: Spike Train Highlighting Difference A - B
threshold = 0.3 * np.max(np.abs(pattern_diff))
highlight = np.abs(pattern_diff) > threshold
highlight_times = t[highlight]

# Extract highlighted spikes
rounded_highlight = np.round(highlight_times, 3)

for i, spikes in enumerate(spike_trains):
    r_spikes = np.round(spikes, 3)
    diff_mask = np.isin(r_spikes, rounded_highlight)
    diff_spikes = spikes[diff_mask]
    base_spikes = spikes[~diff_mask]

    axes[1, 1].vlines(base_spikes, i + 0.5, i + 1.5, color="gray", alpha=0.4)
    axes[1, 1].vlines(diff_spikes, i + 0.5, i + 1.5, color="gold", linewidth=1.5)

axes[1, 1].set_ylim(0.5, n_neurons + 0.5)
axes[1, 1].set_title("Spike Train (Pattern A - Pattern B Highlighted)")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].grid(True, linestyle=':')

plt.tight_layout()

fname = "./images/synthetic_population_difference_plot.png"
plt.savefig(fname, dpi=300)

fname
