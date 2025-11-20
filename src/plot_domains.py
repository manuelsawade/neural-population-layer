import numpy as np
import matplotlib.pyplot as plt

# -------- Generate biologically-inspired tuning curves --------

x = np.linspace(-3, 3, 400)

# Gaussian tuning curve (neural selectivity peak)
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

tuning_curve = gaussian(x, 0, 0.8)
enum = iter("abcdefg")
# Mexican hat (difference-of-Gaussians: excitatory center, inhibitory surround)
def mexican_hat(x, sigma_exc=0.5, sigma_inh=1.5, w_exc=1.0, w_inh=0.6):
    exc = w_exc * np.exp(-0.5 * (x / sigma_exc)**2)
    inh = w_inh * np.exp(-0.5 * (x / sigma_inh)**2)
    return exc - inh

mex_curve = mexican_hat(x)

# Add slight noise for biological realism
tuning_curve_noisy = tuning_curve + np.random.normal(0, 0.02, len(x))
mex_curve_noisy = mex_curve + np.random.normal(0, 0.02, len(x))

# -------- Plot --------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Gaussian tuning curve
axes[0].plot(x, tuning_curve_noisy, color="orange", linewidth=2)
axes[0].set_title("Gaussian Tuning Curve")
axes[0].set_xlabel("Stimulus feature")
axes[0].set_ylabel("Firing rate (a.u.)")
axes[0].grid(True, linestyle=':', alpha=0.5)
axes[0].text(
    0.015,        # a little left of the axes
    0.94,               # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=axes[0].transAxes
)

# Right: Mexican Hat (center-surround receptive field)
axes[1].plot(x, mex_curve_noisy, color="purple", linewidth=2)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Mexican Hat")
axes[1].set_xlabel("Stimulus feature")
axes[1].grid(True, linestyle=':', alpha=0.5)
axes[1].text(
    0.015,        # a little left of the axes
    0.94,                # same vertical height as the title
    f"{next(enum)})",
    fontsize=11, fontweight="bold",
    transform=axes[1].transAxes
)

plt.tight_layout()

fname = "population_code_domains.png"
plt.savefig(fname, dpi=300)
#



# plt.tight_layout()
# plt.savefig("population_code_domains.png")
# plt.close()
