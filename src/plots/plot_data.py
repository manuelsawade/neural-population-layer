from matplotlib import pyplot as plt
import numpy as np

from datasets.mnist import MNIST
import library




# fig = plt.figure(figsize=(8, 8))
# columns = 4
# rows = 1
# for i in range(1, columns*rows +1):
#     image, label = training_data[0]
#     sub = fig.add_subfigure(gridspec[col])
#     sub.imshow(image)
#     sub.title(f'Label: {label}')
# plt.show()

count = 5

fig, axs = plt.subplots(3, 5, figsize=(8, 4), sharex=True, sharey=True)
fig.supylabel('Noise Level')
fig.suptitle(f"MNIST Dataset With Different Noise Levels", y=0.97)
for ax_row, noise in zip(axs, [0.0, 0.5, 1.0]):
    training_data, test_data = MNIST()(training_noise=noise)
    ax_row[0].set_ylabel(f"{noise}",rotation=0, labelpad=15, fontsize=12)
    ax_row[0].yaxis.set_label_coords(-0.4,0.4)

    for i, ax in enumerate(ax_row):   
        image, label = training_data[i]
        ax.imshow(image.squeeze().numpy(), cmap='gray')
        if noise is 0.0:
            ax.set_title(f'Label: {label}')

        ax.set_yticks([])
        ax.set_xticks([])

plt.tight_layout()
plt.savefig(library.get_target_image(__file__), dpi=300)
