import matplotlib.pyplot as plt
import numpy as np

import library

def _get_display_indices(layer_size, threshold):
    """Return list of indices to draw and boolean indicating truncated."""
    if layer_size <= threshold:
        return list(range(layer_size)), False
    # keep first 3 and last 3 (or fewer if threshold small)
    keep_each_side = max(1, threshold // 4)  # small heuristic
    first = list(range(keep_each_side))
    last = list(range(layer_size - keep_each_side, layer_size))
    # ensure uniqueness and sorted
    indices = sorted(set(first + last))
    return indices, True

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, threshold=20):
    """
    Draw a neural network cartoon using matplotlib.
    If a layer has more nodes than `threshold`, most are collapsed and replaced
    with a '[...]' marker. Visible nodes are evenly spaced across that layer's span.
    """
    n_layers = len(layer_sizes)
    # base vertical spacing so largest layer occupies (top-bottom)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(n_layers - 1)

    # Precompute layer vertical spans (top y and bottom y for each layer)
    layer_tops = []
    layer_bottoms = []
    for layer_size in layer_sizes:
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        layer_bottom = layer_top - (layer_size - 1) * v_spacing
        layer_tops.append(layer_top)
        layer_bottoms.append(layer_bottom)

    # Store visible node positions by layer for edge drawing
    layer_node_ys = []

    # Draw nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = layer_tops[n]
        layer_bottom = layer_bottoms[n]

        display_indices, truncated = _get_display_indices(layer_size, threshold)
        # Evenly space the displayed nodes across the full layer span
        if len(display_indices) == 1:
            ys = [0.5 * (layer_top + layer_bottom)]
        else:
            ys = list(np.linspace(layer_top, layer_bottom, num=len(display_indices)))

        x = n * h_spacing + left
        radius = min(v_spacing, h_spacing) * 0.15  # make radius responsive
        for y in ys:
            circle = plt.Circle((x, y), radius, color='w', ec='k', zorder=4)
            ax.add_artist(circle)

        # If truncated, place [...] at mid-point between first and last displayed node
        if truncated:
            y_middle = 0.5 * (ys[0] + ys[-1])
            ax.text(x, y_middle, '[...]', ha='center', va='center', fontsize=10)

        layer_node_ys.append((x, ys, truncated, layer_top, layer_bottom, layer_size))

    # Draw edges between displayed nodes only (connect displayed positions)
    for n in range(len(layer_sizes) - 1):
        x_a, ys_a, trunc_a, top_a, bot_a, size_a = layer_node_ys[n]
        x_b, ys_b, trunc_b, top_b, bot_b, size_b = layer_node_ys[n + 1]

        # connect each displayed node in A to each displayed node in B
        for y_a in ys_a:
            for y_b in ys_b:
                line = plt.Line2D([x_a, x_b], [y_a, y_b], c='k', lw=0.5, alpha=0.6)
                ax.add_artist(line)

        # Optionally, draw a faint connector to the omitted middle region (visual cue)
        # (Not necessary but can help indicate omitted bulk)
        if trunc_a or trunc_b:
            # draw small vertical short ticks near the [...] to hint omitted many-to-many
            if trunc_a:
                # tick at the a-layer [...] position
                _, _, _, _, _, _ = layer_node_ys[n]
            if trunc_b:
                pass  # no extra required

    ax.set_xlim(left - h_spacing*0.5, right + h_spacing*0.5)
    ax.set_ylim(bottom - v_spacing*0.5, top + v_spacing*0.5)
    ax.axis('off')


# Example usage:
if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    draw_neural_net(ax, .1, .9, .1, .9, [784, 128, 10], threshold=10)
    plt.savefig(library.get_target_image(__file__))
