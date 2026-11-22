"""Focused C6 tests for confirmation-annotated heatmap exports."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from tqdm import gui

from LabGym.comparative_transition_heatmap import (
    build_comparative_transition_heatmap,
    export_comparative_transition_heatmap,
)


@pytest.fixture
def comparison():
    behaviors = ['c', 'a', 'b']
    labels = ['a', 'b', 'c']
    a = pd.DataFrame([[np.nan, .8, 0], [.2, np.nan, .8],
                      [np.nan, 1, np.nan]], index=labels, columns=labels)
    b = pd.DataFrame([[np.nan, .3, .7], [.7, np.nan, .3],
                      [0, 1, np.nan]], index=labels, columns=labels)
    # Deliberately supplied B-A matrix, with a sentinel differing from
    # recomputed values, to detect any independent recalculation.
    difference = b - a
    difference.loc['a', 'b'] = -.42
    matrices = [('Custom A', a), ('Custom B', b),
                ('Difference (B - A)', difference)]
    return behaviors, matrices, [('a', 'b'), ('c', 'a')]


@pytest.fixture
def plotted(comparison):
    fig = build_comparative_transition_heatmap(*comparison, dpi=80)
    yield fig, [ax for ax in fig.axes if ax.images]
    fig.clear()


def test_alignment(comparison, plotted):
    behaviors, matrices, _ = comparison
    _, axes = plotted
    assert len(axes) == 3
    for ax, (_, matrix) in zip(axes, matrices):
        assert [t.get_text() for t in ax.get_xticklabels()] == behaviors
        assert [t.get_text() for t in ax.get_yticklabels()] == behaviors
        expected = matrix.loc[behaviors, behaviors].to_numpy()
        np.testing.assert_allclose(ax.images[0].get_array().filled(np.nan),
                                   expected, equal_nan=True)


def test_zero_vs_undefined(plotted):
    _, axes = plotted
    image = axes[0].images[0]
    values = image.get_array()
    # c -> a is undefined; a -> c is a defined zero.
    assert np.ma.getmaskarray(values)[0, 1]
    assert not np.ma.getmaskarray(values)[1, 0]
    assert values[1, 0] == 0
    assert image.cmap.get_bad() == pytest.approx(to_rgba('#dddddd'))
    assert not np.allclose(image.cmap(image.norm(0)), image.cmap.get_bad())
    diff = axes[2].images[0].get_array()
    assert diff[0, 2] == 0
    assert np.ma.getmaskarray(diff)[0, 1]


def test_confirmed_outlines(comparison, plotted):
    behaviors, _, pairs = comparison
    _, axes = plotted
    expected = {(behaviors.index(t)-.5, behaviors.index(s)-.5) for s, t in pairs}
    for ax in axes:
        assert len(ax.patches) == len(pairs)
        assert {p.get_xy() for p in ax.patches} == expected
        for patch in ax.patches:
            assert patch.get_width() == patch.get_height() == 1
            assert not patch.get_fill()
            assert patch.get_edgecolor() == pytest.approx(to_rgba('red'))


def test_difference_direction_and_labels(plotted):
    fig, axes = plotted
    assert [ax.get_title() for ax in axes] == [
        'Custom A', 'Custom B', 'Difference (B - A)']
    image = axes[2].images[0]
    assert image.get_array()[1, 2] == pytest.approx(-.42)
    assert image.get_array()[1, 0] == pytest.approx(.7)
    assert image.norm(0) == pytest.approx(.5)
    assert image.norm.vmin == pytest.approx(-image.norm.vmax)
    assert any(ax.get_ylabel() == 'Difference (B - A)' for ax in fig.axes)
    for ax in axes[:2]:
        assert ax.images[0].get_clim() == (0, 1)


def test_empty_confirmation(comparison, tmp_path):
    behaviors, matrices, _ = comparison
    assert build_comparative_transition_heatmap(behaviors, matrices, []) is None
    output = tmp_path / 'empty'
    assert export_comparative_transition_heatmap(behaviors, matrices, [], output) is None
    assert not output.exists()


def test_png_export_and_no_mutation(comparison, tmp_path):
    behaviors, matrices, pairs = comparison
    originals = [matrix.copy(deep=True) for _, matrix in matrices]
    result = export_comparative_transition_heatmap(*comparison, tmp_path, dpi=80)
    assert result == tmp_path / 'selected_transition_heatmaps.png'
    assert result.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    for (_, matrix), original in zip(matrices, originals):
        pd.testing.assert_frame_equal(matrix, original)


@pytest.mark.parametrize('confirmed', [False, True])
def test_gui_confirmation_preserves_c4(comparison, tmp_path, monkeypatch, confirmed):
    """Exercise the actual confirmation handler without opening a window."""
    from types import SimpleNamespace
    pytest.importorskip('wx')
    import wx
    import wx.grid
    from LabGym import gui_analyzer as gui
    from LabGym import comparative_transition_heatmap as heatmap

    behaviors, matrices, pairs = comparison
    a, b, diff = [matrix for _, matrix in matrices]
    data = dict(behaviors=behaviors, groups=('A', 'B'),
                probability_matrices={'A': a, 'B': b}, difference_matrix=diff,
                signed_differences=pd.DataFrame(pairs, columns=[
                    'source_behavior', 'target_behavior']))
    ranked = pd.DataFrame({'group': ['A', 'B']})
    monkeypatch.setattr(gui, 'confirmed_group_top_transitions', Mock(return_value=ranked))
    monkeypatch.setattr(gui, 'format_comparative_top_transitions', Mock(return_value='Top 5'))
    dialog = Mock()
    dialog.ShowModal.return_value = wx.ID_OK
    for name in ('BoxSizer', 'StaticText', 'Notebook', 'Panel', 'Button',
                 'StdDialogButtonSizer'):
        monkeypatch.setattr(wx, name, Mock())
    monkeypatch.setattr(wx, 'Dialog', Mock(return_value=dialog))
    monkeypatch.setattr(wx.grid, 'Grid', Mock())
    destination = tmp_path / 'selected_transition_heatmaps.png'
    exporter = Mock(return_value=destination)

    monkeypatch.setattr(
        heatmap,
        'export_comparative_transition_heatmap',
        exporter
    )

    monkeypatch.setattr(
        gui,
        'export_comparative_transition_heatmap',
        exporter,
        raising=False
    )
    # A real text buffer detects a saved-path message overwritten by SetValue.
    summary = SimpleNamespace(value='')
    summary.SetValue = lambda value: setattr(summary, 'value', value)
    summary.AppendText = lambda value: setattr(summary, 'value', summary.value + value)
    fake = SimpleNamespace(comparison=data, confirmed_pairs=tuple(pairs) if confirmed else (),
                           output_folder=str(tmp_path), statistics_summary=object(),
                           summary=summary, selection_label=Mock(), update_actions=Mock(),
                           run_statistics=Mock())
    gui.PanelLv2_ComparativeTransitionAnalysis.select_transitions(fake, None)
    fake.run_statistics.assert_not_called()
    fake.update_actions.assert_called_once()
    assert fake.statistics_summary is None
    assert fake.comparison is data
    dialog.Destroy.assert_called_once()
    if confirmed:
        exporter.assert_called_once()
        kwargs = exporter.call_args.kwargs
        assert kwargs['confirmed_pairs'] == tuple(sorted(pairs))
        assert kwargs['behaviors'] == behaviors
        assert kwargs['matrices'][2][0] == 'Difference (A - B)'
        assert kwargs['matrices'][2][1] is diff
        assert str(destination) in summary.value
    else:
        exporter.assert_not_called()
        assert 'No transitions confirmed' in summary.value
        assert not destination.exists()
