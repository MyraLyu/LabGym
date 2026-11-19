"""Export the three-panel comparative transition heatmap after GUI confirmation."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def export_comparative_transition_heatmap(
    comparison, confirmed_pairs, output_folder,
    filename='selected_transition_heatmaps.png',
    group_titles=('Control', 'Treatment'), dpi=200,
):
    """Plot full group matrices; outline *only* user-confirmed directed edges.

    The displayed group matrices are not filtered to the confirmed edges;
    the red outlines indicate the confirmed subset, matching the reference.
    Rows are source behaviors and columns are target behaviors.
    """
    if comparison is None:
        raise ValueError('Generate a comparison before exporting a heatmap.')
    if confirmed_pairs is None:
        raise ValueError('Confirm transitions before exporting a heatmap.')
    behaviors = list(comparison['behaviors'])
    if not behaviors:
        raise ValueError('No behaviors to plot.')
    groups = tuple(comparison.get('groups', ('A', 'B')))
    if len(groups) != 2:
        raise ValueError('Exactly two groups are required.')

    def aligned(matrix):
        if isinstance(matrix, pd.DataFrame):
            return matrix.reindex(index=behaviors, columns=behaviors).to_numpy(dtype=float)
        values = np.asarray(matrix, dtype=float)
        if values.shape != (len(behaviors), len(behaviors)):
            raise ValueError(f'Expected {(len(behaviors), len(behaviors))}, got {values.shape}')
        return values.copy()

    a = aligned(comparison['probability_matrices'][groups[0]])
    b = aligned(comparison['probability_matrices'][groups[1]])
    diff = a - b
    n = len(behaviors)
    selected = set(map(tuple, confirmed_pairs))
    masked = []
    for values in (a, b, diff):
        values = values.copy()
        np.fill_diagonal(values, np.nan)
        masked.append(np.ma.masked_invalid(values))

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 7.6), dpi=dpi)
    fig.subplots_adjust(left=.10, right=.965, bottom=.20, top=.87, wspace=.62)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    delta_cmap = plt.get_cmap('RdBu_r').copy()
    delta_cmap.set_bad('#dddddd')
    max_abs = np.nanmax(np.abs(diff)) if np.any(np.isfinite(diff)) else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1e-12

    for ax, data, title, cm, norm in zip(
        axes, masked,
        (group_titles[0], group_titles[1], f'{group_titles[0]} - {group_titles[1]}'),
        (cmap, cmap, delta_cmap),
        (plt.Normalize(0, 1), plt.Normalize(0, 1), TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)),
    ):
        image = ax.imshow(data, cmap=cm, norm=norm, interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=16, pad=8)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(behaviors, rotation=45, ha='right', rotation_mode='anchor', fontsize=11)
        ax.set_yticklabels(behaviors, fontsize=11)
        ax.set_xlabel('Target behavior', fontsize=13)
        ax.set_ylabel('Source behavior', fontsize=13)
        ax.set_xlim(-.5, n-.5)
        ax.set_ylim(n-.5, -.5)
        for i, source in enumerate(behaviors):
            for j, target in enumerate(behaviors):
                if i != j and (source, target) in selected:
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                           edgecolor='red', linewidth=1.6))
        cb = fig.colorbar(image, ax=ax, fraction=.047, pad=.045)
        if cm is delta_cmap:
            cb.set_label(f'{group_titles[0]} - {group_titles[1]}', fontsize=12)
        else:
            cb.set_ticks(np.linspace(0, 1, 6))

    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    fig.savefig(destination, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return destination
