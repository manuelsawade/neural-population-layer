import numpy as np
import matplotlib.pyplot as plt

# 1. Gaussian tuning curve
x = np.random.normal(0, 1, 200)
sigma = 1.0
gaussian = np.exp(-x**2 / (2 * sigma**2))

plt.figure()
plt.plot(x, gaussian, color="cornflowerblue")
plt.scatter(x, gaussian, label="Gaussian")
plt.grid(True)
plt.savefig("gaussian_normal.png")
plt.close()

def cosine(x, n, min, max):
    y = np.cos(np.linspace(0, np.pi, n))
    y = min + (y - y.min()) * (max - min) / (y.max() - y.min())
    p = np.repeat(x, n)
    sigma = 1.0
    return (y, np.exp(-0.5 * ((p - y) / sigma) ** 2) / (2 * sigma ** 2))

# y1, gaussian1 = cosine(-1.5, 200, -3, 3)
# y2, gaussian2 = cosine(-1, 200, -3, 3)
# y3, gaussian3 = cosine(-0.5, 200, -3, 3)
y4, gaussian4 = cosine(0, 200, -3, 3)
# y5, gaussian5 = cosine(0.5, 200, -3, 3)
# y6, gaussian6 = cosine(1.0, 200, -3, 3)
# y7, gaussian7 = cosine(1.5, 200, -3, 3)

plt.figure()
# plt.plot(y1, gaussian1, label="Gaussian", color="r")
# plt.plot(y2, gaussian2, label="Gaussian", color="b")
# plt.plot(y3, gaussian3, label="Gaussian", color="g")
plt.plot(y4, gaussian4, label="Gaussian", color="c")
# plt.plot(y5, gaussian5, label="Gaussian", color="m")
# plt.plot(y6, gaussian6, label="Gaussian", color="y")
# plt.plot(y7, gaussian7, label="Gaussian", color="k")

plt.grid(True)
plt.savefig("gaussian_curve_cosine.png")
plt.close()

# 2. Mexican hat curve (Ricker wavelet approximation)
mexican_hat = (1 - (x**2 / sigma**2)) * np.exp(-x**2 / (2 * sigma**2))

plt.figure()
plt.plot(x, mexican_hat, label="Mexican Hat", color="orange")
plt.grid(True)
plt.savefig("mexican_hat_curve.png")
plt.close()

# 3. Sine curve with frequency 8
t = np.linspace(0, 1, 500)
frequency = 4
sine_curve = np.sin(2 * np.pi * frequency * t)

plt.figure()
plt.plot(t, sine_curve, label="Sine Wave (f=8Hz)", color="green")
plt.grid(True)
plt.savefig("sine_curve.png")
plt.close()

# 4. Log Normal

x = np.linspace(-3, 3, 200)
sigma = 1.0

coeff = 1.0 / (x * sigma * np.sqrt(2.0 * np.pi))
exponent = -((np.log(x)) ** 2) / (2 * sigma**2)
log_normal = coeff * np.exp(exponent)

plt.figure()
plt.plot(x, log_normal, label="LogNormal", color="purple")
plt.grid(True)
plt.savefig("log_normal.png")
plt.close()


# 5. circular encoder
xmin = -3
xmax = 3

x1 = np.repeat(-1.5, 200)
x2 = np.repeat(0, 200)
x3 = np.repeat(1.5, 200)
x4 = np.repeat(3, 200)

mu = np.linspace(xmin, xmax, 200)
sigma = 0.85
L = xmax - xmin

# expand dims for broadcasting
# stim: [B, N, 1] → [B, N, 1]
# pref: [N, P]   → [1, N, P]
# [B, N, P]

dist1 = np.abs(x1 - mu)
dist1 = np.minimum(dist1, L - dist1)
resp1 = np.exp(-0.5 * (dist1 / sigma) ** 2)

dist2 = np.abs(x2 - mu)
dist2 = np.minimum(dist2, L - dist2)
resp2 = np.exp(-0.5 * (dist2 / sigma) ** 2)

dist3 = np.abs(x3 - mu)
dist3 = np.minimum(dist3, L - dist3)
resp3 = np.exp(-0.5 * (dist3 / sigma) ** 2)

dist4 = np.abs(x4 - mu)
dist4 = np.minimum(dist4, L - dist4)
resp4 = np.exp(-0.5 * (dist4 / sigma) ** 2)

plt.figure()
plt.plot(mu, resp1, label="Gaussian Circular", color="cornflowerblue")
plt.plot(mu, resp2, label="Gaussian Circular", color="royalblue")
plt.plot(mu, resp3, label="Gaussian Circular", color="cornflowerblue")
plt.plot(mu, resp4, label="Gaussian Circular", color="lightsteelblue")
plt.grid(True)
plt.savefig("gaussian_circular.png")
plt.close()
