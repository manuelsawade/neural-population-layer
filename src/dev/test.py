import torch

from population_layer import PopulationCodedLayer

torch.set_printoptions(precision=10)

x = torch.tensor([[0.4, 0.3, 0.9, 0.3, 0.2]])
y = torch.randn(4, 5)
layer = PopulationCodedLayer(input_dim=5, hidden_dim=5, debug=True)

output = layer(x)



image, label = training_data[0]  # image is a tensor

# Plot the image
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()