import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_labels=None, simplified=False):
    """
    Draw a neural network diagram using matplotlib.
    """
    v_spacing = (top - bottom)/float(max(layer_sizes if not simplified else [max(10, min(50, n)) for n in layer_sizes]))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    node_positions = []
    for n, layer_size in enumerate(layer_sizes):
        # Limit the number of visible nodes for full version rendering speed
        visible_size = layer_size if (not simplified and layer_size < 400) else min(layer_size, 50)
        layer_top = v_spacing*(visible_size - 1)/2. + (top + bottom)/2.
        positions = [(n*h_spacing + left, layer_top - m*v_spacing) for m in range(visible_size)]
        node_positions.append(positions)

    # Connections (reduced for clarity and speed)
    for n, (layer1, layer2) in enumerate(zip(node_positions[:-1], node_positions[1:])):
        step = 1 if simplified else max(1, len(layer1)//50)
        for i, (x1, y1) in enumerate(layer1[::step]):
            for j, (x2, y2) in enumerate(layer2[::step]):
                ax.plot([x1, x2], [y1, y2], 'k-', lw=0.3 if not simplified else 0.7, alpha=0.5)

    for layer, positions in enumerate(node_positions):
        for (x, y) in positions:
            ax.plot(x, y, 'o', markersize=3 if not simplified else 6, color='royalblue')
        if layer_labels:
            ax.text(x, top + 0.05, layer_labels[layer], fontsize=10, ha='center')

# Create two plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# "Full" version (still reduced for practicality)
ax1 = axes[0]
draw_neural_net(ax1, 0.1, 0.9, 0.1, 0.9, [784, 200, 10],
                layer_labels=['Input (784)', 'Hidden (200)', 'Output (10)'], simplified=False)
ax1.set_title("Full Neural Network (784-200-10)", fontsize=12)
ax1.axis('off')

# Simplified version
ax2 = axes[1]
draw_neural_net(ax2, 0.1, 0.9, 0.1, 0.9, [784, 200, 10],
                layer_labels=['Input (784)', 'Hidden (200)', 'Output (10)'], simplified=True)
ax2.set_title("Simplified Neural Network (784-200-10)", fontsize=12)
ax2.axis('off')

plt.tight_layout()
plt.show()

