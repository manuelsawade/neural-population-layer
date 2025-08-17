from matplotlib import pyplot as plt

from data.mnist import MNIST


training_data, test_data = MNIST()(training_noise=0.0)
image, label = training_data[0]

plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()