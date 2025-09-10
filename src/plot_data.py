from matplotlib import pyplot as plt
import numpy as np

from datasets.mnist import MNIST



training_data, test_data = MNIST()(training_noise=0.6)

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

fig, axs = plt.subplots(1, 5, figsize=(10, 3))
for ax, i in zip(axs, range(0, 5)):
    image, label = training_data[i]
    ax.imshow(image.squeeze().numpy(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.show()
