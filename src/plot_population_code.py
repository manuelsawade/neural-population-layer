import torch
import matplotlib.pyplot as plt
import numpy as np

N = 10
SIGMA = 1.2
PREFERRED_VALS = torch.arange(N, dtype=torch.float32)

SELECTED_INDEX = 4

print(torch.normal(
                    mean=0, 
                    std=1, 
                    size=(2, 12)))

def population_encode(digit, preferred_vals=PREFERRED_VALS, sigma=SIGMA):
    activations = torch.exp(-0.5 * ((digit - preferred_vals) / sigma) ** 2)
    return activations

def population_decode(population_code, preferred_vals=PREFERRED_VALS):
    decoded_digits = (population_code * preferred_vals).sum(dim=1) / population_code.sum(dim=1)
    return decoded_digits

true_digits = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32)
population_codes = torch.stack([population_encode(d) for d in true_digits])

noisy_population_codes = population_codes + 0.1 * torch.randn_like(population_codes)

decoded_digits = population_decode(noisy_population_codes)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(PREFERRED_VALS, population_codes[SELECTED_INDEX].numpy())
axes[0].set_title(f"Population code for digit {int(true_digits[SELECTED_INDEX].item())}")
axes[0].set_xlabel("Preferred digit")
axes[0].set_ylabel("Activation")

axes[1].scatter(true_digits, decoded_digits, color='red')
for i in range(len(true_digits)):
    axes[1].text(true_digits[i] + 0.1, decoded_digits[i], f"{decoded_digits[i]:.2f}")
axes[1].plot([0, 9], [0, 9], 'k--', alpha=0.5)
axes[1].set_title("True vs Decoded digits")
axes[1].set_xlabel("True digit")
axes[1].set_ylabel("Decoded digit")

plt.tight_layout()
plt.savefig("encode_decode.png")
