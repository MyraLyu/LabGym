"""Export confirmation-annotated comparative transition heatmaps."""

from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib import colormaps


def build_comparative_transition_heatmap(
    behaviors,
    matrices,
    confirmed_pairs,
    dpi=200,
):
    """Build a heatmap using the exact matrices displayed in the GUI."""

    confirmed_pairs = tuple(confirmed_pairs or ())

    if not confirmed_pairs:
        return None

    behaviors = list(behaviors)

    if not behaviors:
        raise ValueError('No behaviors to plot.')

    if len(matrices) != 3:
        raise ValueError(
            'Expected two group matrices and one difference matrix.'
        )

    n = len(behaviors)
    selected = set(map(tuple, confirmed_pairs))

    def aligned(matrix):
        if isinstance(matrix, pd.DataFrame):
            return matrix.loc[
                behaviors, behaviors
            ].to_numpy(dtype=float)

        values = np.asarray(matrix, dtype=float)

        if values.shape != (n, n):
            raise ValueError(
                f'Expected {(n, n)}, got {values.shape}'
            )

        return values.copy()

    # Preserve GUI values without recalculating the difference.
    titles = [title for title, _ in matrices]

    values = [
        np.ma.masked_invalid(aligned(matrix))
        for _, matrix in matrices
    ]

    # Keep true zero values separate from undefined values.
    diff = values[2]
    finite_diff = diff.compressed()

    max_abs = (
        float(np.max(np.abs(finite_diff)))
        if finite_diff.size
        else 0.0
    )

    if max_abs == 0:
        max_abs = 1e-12

    probability_cmap = colormaps['viridis'].copy()
    probability_cmap.set_bad('#dddddd')

    difference_cmap = colormaps['RdBu_r'].copy()
    difference_cmap.set_bad('#dddddd')

    norms = (
        Normalize(vmin=0, vmax=1),
        Normalize(vmin=0, vmax=1),
        TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs,
        ),
    )

    cmaps = (
        probability_cmap,
        probability_cmap,
        difference_cmap,
    )

    # Use a non-interactive canvas without changing the
    # application's global Matplotlib backend.
    fig = Figure(figsize=(19.2, 7.6), dpi=dpi)
    FigureCanvasAgg(fig)

    axes = fig.subplots(1, 3)

    fig.subplots_adjust(
        left=.10,
        right=.965,
        bottom=.20,
        top=.87,
        wspace=.62,
    )

    for index, ax in enumerate(axes):

        image = ax.imshow(
            values[index],
            cmap=cmaps[index],
            norm=norms[index],
            interpolation='nearest',
            aspect='equal',
        )

        ax.set_title(
            titles[index],
            fontsize=16,
            pad=8,
        )

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))

        ax.set_xticklabels(
            behaviors,
            rotation=45,
            ha='right',
            rotation_mode='anchor',
            fontsize=11,
        )

        ax.set_yticklabels(
            behaviors,
            fontsize=11,
        )

        ax.set_xlabel(
            'Target behavior',
            fontsize=13,
        )

        ax.set_ylabel(
            'Source behavior',
            fontsize=13,
        )

        ax.set_xlim(-.5, n - .5)
        ax.set_ylim(n - .5, -.5)

        # Outline only confirmed directed transitions.
        for i, source in enumerate(behaviors):
            for j, target in enumerate(behaviors):

                if (source, target) in selected:

                    ax.add_patch(
                        Rectangle(
                            (j - .5, i - .5),
                            1,
                            1,
                            fill=False,
                            edgecolor='red',
                            linewidth=1.6,
                        )
                    )

        cb = fig.colorbar(
            image,
            ax=ax,
            fraction=.047,
            pad=.045,
        )

        if index == 2:
            # Use the exact difference-direction label from GUI.
            cb.set_label(
                titles[index],
                fontsize=12,
            )

        else:
            cb.set_ticks(
                np.linspace(0, 1, 6)
            )

    return fig


def export_comparative_transition_heatmap(
    behaviors,
    matrices,
    confirmed_pairs,
    output_folder,
    filename='selected_transition_heatmaps.png',
    dpi=200,
):
    """Save the existing GUI comparison matrices as an annotated PNG."""

    confirmed_pairs = tuple(confirmed_pairs or ())

    # Do not create any output for empty confirmation.
    if not confirmed_pairs:
        return None

    fig = build_comparative_transition_heatmap(
        behaviors=behaviors,
        matrices=matrices,
        confirmed_pairs=confirmed_pairs,
        dpi=dpi,
    )

    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    destination = output / filename

    try:
        fig.savefig(
            destination,
            dpi=dpi,
            facecolor='white',
            bbox_inches='tight',
        )

    finally:
        fig.clear()

    return destination