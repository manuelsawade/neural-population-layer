from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Synthetic Data Generation With Biological Realism ----------------
enum = iter("abcdefg")
np.random.seed(42)

T = 2.0    # seconds
fs = 100  # sampling rate
t = np.linspace(0, T, int(T * fs))

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

centers_A = np.linspace(0.3, 1.7, 5) + np.random.normal(0, 0.05, 5)
centers_B = np.linspace(0.2, 1.8, 6) + np.random.normal(0, 0.05, 6)

pattern_A = np.zeros_like(t)
pattern_B = np.zeros_like(t)

for c in centers_A:
    pattern_A += gaussian(t, c, 0.08 + np.random.uniform(0.01, 0.05))

for c in centers_B:
    pattern_B += gaussian(t, c, 0.08 + np.random.uniform(0.01, 0.05))

pattern_A += np.random.normal(0, 0.05, len(t))
pattern_B += np.random.normal(0, 0.05, len(t))

pattern_A = (pattern_A - pattern_A.min()) / (pattern_A.max() - pattern_A.min()) * 3
pattern_B = (pattern_B - pattern_B.min()) / (pattern_B.max() - pattern_B.min()) * 3

n_neurons = 40
spike_trains_A = []
spike_trains_B = []
latencies = []

peak_idx = np.argmax(pattern_A)
peak_time = t[peak_idx]

def enforce_refractory(spikes, refractory_ms=0.003):
    if len(spikes) == 0:
        return spikes
    cleaned = [spikes[0]]
    for s in spikes[1:]:
        if s - cleaned[-1] > refractory_ms:
            cleaned.append(s)
    return np.array(cleaned)

for i in range(n_neurons):
    baseline = np.random.uniform(5, 15)
    sens_A = np.random.uniform(0.5, 1.5)
    sens_B = np.random.uniform(0.5, 1.5)
    burst_factor = np.random.choice([1, 2.5], p=[0.7, 0.3])
    
    rate_A = baseline + sens_A * pattern_A
    rate_B = baseline + sens_B * pattern_B

    prob_A = (rate_A / np.max(rate_A)) * 0.01 * burst_factor
    prob_B = (rate_B / np.max(rate_B)) * 0.01 * burst_factor

    spikes_A = t[np.random.rand(len(t)) < prob_A]
    spikes_B = t[np.random.rand(len(t)) < prob_B]

    spikes_A = enforce_refractory(spikes_A)
    spikes_B = enforce_refractory(spikes_B)

    spike_trains_A.append(spikes_A)
    spike_trains_B.append(spikes_B)

    if len(spikes_A) > 0:
        latencies.append(np.mean(spikes_A - peak_time))
    else:
        latencies.append(5)

sorted_idx = np.argsort(latencies)

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(4, 2, height_ratios=[1.25,1, 1.25,1])

# gs = GridSpec(3, 2, figure=fig)
# ax1 = fig.add_subplot(gs[0, :])
# ax2 = fig.add_subplot(gs[1, :])
# # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
# #ax2 = fig.add_subplot(gs[1, :-1])
# ax3 = fig.add_subplot(gs[-1, 0])
# ax4 = fig.add_subplot(gs[-1:, -1])

# Raster A+B
ax_raster_all = fig.add_subplot(gs[0, :])
ax_raster_all.set_title("Spike Train")
for row, idx in enumerate(sorted_idx):
    ax_raster_all.vlines(spike_trains_A[idx], row+0.6, row+1.0, color="tab:blue")
    ax_raster_all.vlines(spike_trains_B[idx], row+0.0, row+0.4, color="tab:red")
ax_raster_all.set_ylim(-0.5, n_neurons + 0.5)
ax_raster_all.set_ylabel("Neuron (sorted)")
ax_raster_all.grid(True, linestyle=':')
ax_raster_all.text(
    0.005,        # a little left of the axes
    0.82,               # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=ax_raster_all.transAxes
)

#fig.add_subplot(gs[0, 1]).axis("off")

# PSTH A+B
bin_edges = np.linspace(0, T, 80)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
all_spikes_A = np.concatenate([spike_trains_A[i] for i in sorted_idx])
all_spikes_B = np.concatenate([spike_trains_B[i] for i in sorted_idx])
hist_A, _ = np.histogram(all_spikes_A, bins=bin_edges)
hist_B, _ = np.histogram(all_spikes_B, bins=bin_edges)

ax_psth_all = fig.add_subplot(gs[1, :])
ax_psth_all.fill_between(bin_centers, hist_A, color="tab:blue", alpha=0.3)
ax_psth_all.fill_between(bin_centers, hist_B, color="tab:red", alpha=0.3)
ax_psth_all.plot(t, pattern_A * (max(hist_A)+2) / max(pattern_A), color="tab:blue")
ax_psth_all.plot(t, pattern_B * (max(hist_B)+2) / max(pattern_B), color="tab:red")
ax_psth_all.set_ylabel("PSTH")
ax_psth_all.grid(True, linestyle=':')
ax_psth_all.text(
    0.005,        # a little left of the axes
    0.80,               # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=ax_psth_all.transAxes
)

fig.add_subplot(gs[1, 1]).axis("off")

# Raster A only
ax_raster_A = fig.add_subplot(gs[2, :])
ax_raster_A.set_title("Spike Train")
for row, idx in enumerate(sorted_idx):
    ax_raster_A.vlines(spike_trains_A[idx], row+0.5, row+1.5, color="tab:blue")
ax_raster_A.set_ylim(0.5, n_neurons + 0.5)
ax_raster_A.set_ylabel("Neuron (sorted)")
ax_raster_A.grid(True, linestyle=':')
ax_raster_A.text(
    0.005,        # a little left of the axes
    0.82,               # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=ax_raster_A.transAxes
)

fig.add_subplot(gs[2, 1]).axis("off")

# PSTH A only
hist_Aonly, _ = np.histogram(all_spikes_A, bins=bin_edges)
ax_psth_A = fig.add_subplot(gs[3, :])
ax_psth_A.fill_between(bin_centers, hist_Aonly, color="tab:blue", alpha=0.3)
ax_psth_A.plot(t, pattern_A * (max(hist_Aonly)+2) / max(pattern_A), color="tab:blue")
ax_psth_A.set_ylabel("PSTH (A only)")
ax_psth_A.grid(True, linestyle=':')
ax_psth_A.text(
    0.005,        # a little left of the axes
    0.80,               # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=ax_psth_A.transAxes
)

fig.add_subplot(gs[3, 1]).axis("off")

# Activity patterns
# ax_A = fig.add_subplot(gs[4:, 0])
# ax_A.plot(t, pattern_A, color="tab:blue")
# ax_A.set_title("Activity Pattern A (Gaussian Tuning Curves)")
# ax_A.set_ylabel("Activity")
# ax_A.set_xlabel("Time (s)")
# ax_A.grid(True, linestyle=':')

# ax_B = fig.add_subplot(gs[4:, 1])
# ax_B.plot(t, pattern_B, color="tab:red")
# ax_B.set_title("Activity Pattern B (Gaussian Tuning Curves)")
# ax_B.set_xlabel("Time (s)")
# ax_B.grid(True, linestyle=':')

plt.tight_layout()

fname = "spike.png"
plt.savefig(fname, dpi=300)

fname
