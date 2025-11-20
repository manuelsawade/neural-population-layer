from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Synthetic Data Generation With Biological Realism ----------------


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

np.random.seed(42)

T = 2.0    # seconds
fs = 3000  # sampling rate
t = np.linspace(0, T, int(T * fs))

# -------- Activity patterns generated as overlapping Gaussian tuning curves --------
enum = iter("abcdefg")
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

# Create several Gaussian bumps for each pattern
centers_A = np.linspace(0.3, 1.7, 5) + np.random.normal(0, 0.05, 5)
centers_B = np.linspace(0.2, 1.8, 6) + np.random.normal(0, 0.05, 6)

pattern_A = np.zeros_like(t)
pattern_B = np.zeros_like(t)

for c in centers_A:
    pattern_A += gaussian(t, c, 0.08 + np.random.uniform(0.01, 0.05))

for c in centers_B:
    pattern_B += gaussian(t, c, 0.08 + np.random.uniform(0.01, 0.05))

# Add noise for realism
pattern_A += np.random.normal(0, 0.05, len(t))
pattern_B += np.random.normal(0, 0.05, len(t))

# Normalize to amplitude ~3
pattern_A = (pattern_A - pattern_A.min()) / (pattern_A.max() - pattern_A.min()) * 3
pattern_B = (pattern_B - pattern_B.min()) / (pattern_B.max() - pattern_B.min()) * 3

n_neurons = 40
spike_trains_A = []
spike_trains_B = []
latencies = []

# Peak of pattern A for latency sorting
peak_idx = np.argmax(pattern_A)
peak_time = t[peak_idx]

# Spike generation with biological realism
for i in range(n_neurons):
    baseline = np.random.uniform(5, 15)  # heterogeneous baseline rates
    
    sens_A = np.random.uniform(0.5, 1.5)
    sens_B = np.random.uniform(0.5, 1.5)
    burst_factor = np.random.choice([1, 2.5], p=[0.7, 0.3])
    
    rate_A = baseline + sens_A * pattern_A
    rate_B = baseline + sens_B * pattern_B

    prob_A = (rate_A / np.max(rate_A)) * 0.01 * burst_factor
    prob_B = (rate_B / np.max(rate_B)) * 0.01 * burst_factor

    spikes_A = t[np.random.rand(len(t)) < prob_A]
    spikes_B = t[np.random.rand(len(t)) < prob_B]

    # Enforce refractory period
    def enforce_refractory(spikes, refractory_ms=0.003):
        if len(spikes) == 0:
            return spikes
        cleaned = [spikes[0]]
        for s in spikes[1:]:
            if s - cleaned[-1] > refractory_ms:
                cleaned.append(s)
        return np.array(cleaned)

    spikes_A = enforce_refractory(spikes_A)
    spikes_B = enforce_refractory(spikes_B)

    spike_trains_A.append(spikes_A)
    spike_trains_B.append(spikes_B)

    if len(spikes_A) > 0:
        latencies.append(np.mean(spikes_A - peak_time))
    else:
        latencies.append(5)

# Sort neurons by latency
sorted_idx = np.argsort(latencies)

# --------------------- Plotting ----------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 2, 2]})

fig = plt.figure(layout="constrained")

gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
#ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[-1, 0])
ax4 = fig.add_subplot(gs[-1:, -1])

# Row 1: Spike train (all activity)
#ax1, ax1b = axes[0]
ax1.set_title("Spike Train")
for row, idx in enumerate(sorted_idx):
    ax1.vlines(spike_trains_A[idx], row+5.6, row+3.0, color="tab:blue")
    ax1.vlines(spike_trains_B[idx], row+5.0, row+3.4, color="tab:red")
ax1.set_ylim(-0.5, n_neurons + 0.5)
ax1.set_ylabel("Neuron (sorted)")
ax1.grid(True, linestyle=':')
# ax1b.axis("off")

# Row 2: Spike train (A only)
#ax2, ax2b = axes[1]
ax2.set_title("Spikes")
for row, idx in enumerate(sorted_idx):
    ax2.vlines(spike_trains_A[idx], row+5.5, row+3.4, color="tab:blue")
ax2.set_ylim(0.5, n_neurons + 0.5)
ax2.set_ylabel("Neuron (sorted)")
ax2.grid(True, linestyle=':')
#ax2b.axis("off")

# Row 3: Activity Pattern A and B
#sax3, ax4 = axes[2]

# Pattern A
ax3.plot(t, pattern_A, color="tab:blue")
ax3.set_title("Spike Pattern")
ax3.set_ylabel("Activity")
ax3.set_xlabel("Time (s)")
ax3.grid(True, linestyle=':')

# Pattern B
ax4.plot(t, pattern_B, color="tab:red")
ax4.set_title("Spontaneous Activity")
ax4.set_xlabel("Time (s)")
ax4.grid(True, linestyle=':')

plt.tight_layout()

fname = "./images/spike_train_sorted_gaussian_patterns.png"
plt.savefig(fname, dpi=300)

fname
