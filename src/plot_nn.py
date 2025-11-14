import matplotlib.pyplot as plt
import library

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, threshold=10):
    '''
    Draw a neural network cartoon using matplotlib.

    Parameters:
        ax : matplotlib.axes.AxesSubplot
            The axes on which to plot.
        left, right, bottom, top : float
            Coordinates for placement.
        layer_sizes : list of int
            Number of neurons per layer.
        threshold : int
            Max number of nodes/edges to display before collapsing.
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(n_layers - 1)

    # Draw nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        
        # Decide which nodes to show
        if layer_size > threshold:
            indices = [0, 1, 2, layer_size - 3, layer_size - 2, layer_size - 1]
            show_indices = sorted(set(i for i in indices if 0 <= i < layer_size))
        else:
            show_indices = range(layer_size)

        for m in show_indices:
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 4.0,
                color='w',
                ec='k',
                zorder=4
            )
            ax.add_artist(circle)

        # Add [...] label if truncated
        if layer_size > threshold:
            ax.text(
                n * h_spacing + left,
                layer_top - (layer_size / 2.0) * v_spacing,
                '[...]',
                ha='center', va='center', fontsize=10
            )

    # Draw edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0

        # Determine visible nodes for each side
        if layer_size_a > threshold:
            idx_a = [0, 1, 2, layer_size_a - 3, layer_size_a - 2, layer_size_a - 1]
            show_a = sorted(set(i for i in idx_a if 0 <= i < layer_size_a))
        else:
            show_a = range(layer_size_a)

        if layer_size_b > threshold:
            idx_b = [0, 1, 2, layer_size_b - 3, layer_size_b - 2, layer_size_b - 1]
            show_b = sorted(set(i for i in idx_b if 0 <= i < layer_size_b))
        else:
            show_b = range(layer_size_b)

        # Draw selected edges
        for m in show_a:
            for o in show_b:
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    c='k',
                    lw=0.5
                )
                ax.add_artist(line)

    ax.set_xlim(left - h_spacing, right + h_spacing)
    ax.set_ylim(bottom - v_spacing, top + v_spacing)
    ax.axis('off')


# Example usage:
if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    draw_neural_net(ax, .1, .9, .1, .9, [784, 128, 10], threshold=30)
    plt.savefig(library.get_target_image(__file__))
