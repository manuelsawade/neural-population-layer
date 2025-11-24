#!/usr/bin/env python3
"""
plot_thesis_structure.py

Generates a matplotlib visualization of the thesis structure as a labeled grid
with hierarchical grouping (Chapters -> Subsections). This is a static
representation based on the current LaTeX organization.

Usage (from repository root):
    python plot_thesis_structure.py            # show interactively
    python plot_thesis_structure.py --save thesis_structure.png

Options:
    --save <path>   Save the figure instead of (or in addition to) showing it.

The layout algorithm assigns one row per chapter. Subsections are drawn as
adjacent boxes within that row. Empty chapters (e.g., Conclusion, Appendix)
are still rendered with a single placeholder box.
"""
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
@dataclass
class Chapter:
    title: str
    subsections: List[str]

# Thesis structure derived from provided LaTeX files (manual mapping)
THESIS_STRUCTURE: List[Chapter] = [
    Chapter(
        "Introduction",
        ["Motivation", "Objective", "Limitations", "Contributions", "Structure"]
    ),
    Chapter(
        "Theory",
        [
            "Artificial Neural Networks",
            "Activation Functions",
            "Metrics",
            "Population Codes",
            "Methods"
        ],
    ),
    Chapter(
        "Development",
        [
            "Framework",
            "Activation Function Design",
            "Evaluation MNIST",
            "Evaluation CIFAR10",
            "Evaluation Summary",
        ],
    ),
    Chapter("Conclusion", []),
    Chapter("Appendix", []),
]

# ---------------------------------------------------------------------------
# Layout & Rendering
# ---------------------------------------------------------------------------

def compute_layout(structure: List[Chapter]) -> Tuple[Dict[str, Tuple[float, float, float, float]], float, float]:
    """Compute normalized rectangle positions for each chapter and subsection.

    Returns:
        mapping: dict key -> (x, y, w, h) in figure-relative coordinates
        total_width: overall width used (for reference)
        total_height: overall height used (for reference)
    """
    # Layout parameters
    row_height = 1.0  # Each chapter gets equal vertical allocation (will normalize later)
    h_gap = 0.04      # horizontal gap between subsection boxes
    v_gap = 0.06      # vertical gap between chapter rows
    chapter_label_width = 0.15  # width reserved for chapter label box
    min_subsection_width = 0.08 # minimum width for a subsection box

    # Count rows
    n_rows = len(structure)
    total_height = n_rows * row_height + (n_rows - 1) * v_gap

    # Horizontal extent: chapter label + subsections (variable)
    # We compute maximum needed width across chapters.
    chapter_widths = []
    for ch in structure:
        n_sub = max(1, len(ch.subsections))
        subsections_width = n_sub * min_subsection_width + (n_sub - 1) * h_gap
        chapter_total = chapter_label_width + h_gap + subsections_width
        chapter_widths.append(chapter_total)
    total_width = max(chapter_widths)

    layout = {}
    current_y = 0.0

    for ch in structure:
        # Normalize y position from bottom (matplotlib origin bottom-left for patches)
        chapter_y = current_y
        chapter_h = row_height
        n_sub = max(1, len(ch.subsections))
        subsections_width = n_sub * min_subsection_width + (n_sub - 1) * h_gap
        chapter_total = chapter_label_width + h_gap + subsections_width

        # Center chapter content horizontally within total_width
        x_offset = (total_width - chapter_total) / 2.0

        # Chapter label box
        layout[f"Chapter:{ch.title}"] = (
            x_offset,
            chapter_y,
            chapter_label_width,
            chapter_h,
        )

        # Subsection boxes
        if ch.subsections:
            sx = x_offset + chapter_label_width + h_gap
            for i, sub in enumerate(ch.subsections):
                layout[f"Subsection:{ch.title}:{sub}"] = (
                    sx + i * (min_subsection_width + h_gap),
                    chapter_y,
                    min_subsection_width,
                    chapter_h,
                )
        else:
            # Placeholder for empty chapter
            sx = x_offset + chapter_label_width + h_gap
            layout[f"Subsection:{ch.title}:<None>"] = (
                sx,
                chapter_y,
                min_subsection_width,
                chapter_h,
            )

        current_y += row_height + v_gap

    # Normalize to [0,1] coordinate space
    normalized_layout = {}
    for k, (x, y, w, h) in layout.items():
        normalized_layout[k] = (
            x / total_width,
            y / total_height,
            w / total_width,
            h / total_height,
        )

    return normalized_layout, total_width, total_height


def draw_structure(structure: List[Chapter], ax: plt.Axes) -> None:
    layout, _, _ = compute_layout(structure)

    # Style parameters
    chapter_facecolor = "#1f77b4"  # blue
    subsection_facecolor = "#ff7f0e"  # orange
    empty_facecolor = "#d3d3d3"  # light gray for placeholders
    edgecolor = "#333333"

    for key, (x, y, w, h) in layout.items():
        parts = key.split(":")
        kind = parts[0]
        title = parts[1]
        if kind == "Chapter":
            fc = chapter_facecolor
            label = title
        else:
            subsection = parts[2]
            if subsection == "<None>":
                fc = empty_facecolor
                label = "(no subsections)"
            else:
                fc = subsection_facecolor
                label = subsection

        rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=edgecolor, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if fc != empty_facecolor else "#333333",
            wrap=True,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Thesis Structure Overview", fontsize=14, pad=12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot thesis structure grid")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure (PNG/PDF)")
    return parser.parse_args()


def main():
    args = parse_args()
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_structure(THESIS_STRUCTURE, ax)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved thesis structure plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
