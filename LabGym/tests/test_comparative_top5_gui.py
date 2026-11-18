"""GUI-facing Top 5 regression tests (requires wxPython / LabGym GUI dependencies)."""
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

wx = pytest.importorskip('wx')
from LabGym import gui_analyzer as gui
from LabGym import comparative_transition_analysis as backend


def rankings(values):
    return pd.DataFrame([
        dict(rank=i + 1, source_behavior=f'S{i}', target_behavior='T',
             mean_transition_probability=value)
        for i, value in enumerate(values)
    ])


def comparison_for_confirm_refresh():
    """Comparison with three edges; a→b ranks highest if all are confirmed."""
    behaviors = ['a', 'b', 'c']
    edges = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    rows = []
    for group, probabilities in [('A', [0.9, 0.5, 0.0]), ('B', [0.8, 0.4, 0.0])]:
        for (source, target), value in zip(edges, probabilities):
            rows.append(dict(
                group=group,
                source_behavior=source,
                target_behavior=target,
                mean_transition_probability=value,
                n_animals=2,
                n_contributing_transition_probability=2,
            ))
    matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)
    for source, target in edges:
        matrix.loc[source, target] = 0.1
    return {
        'groups': ('A', 'B'),
        'behaviors': behaviors,
        'transition_summary': pd.DataFrame(rows),
        'signed_differences': pd.DataFrame([
            {'source_behavior': source, 'target_behavior': target}
            for source, target in edges
        ]),
        'probability_matrices': {'A': matrix.copy(), 'B': matrix.copy()},
        'difference_matrix': matrix.copy(),
        'per_animal_transitions': pd.DataFrame([
            dict(group='A', source_recording='/data/rec', species='mouse',
                 animal_id=1, source_behavior='b', target_behavior='c',
                 transition_probability=0.5, transition_count=2,
                 fps=30, fps_source='metadata'),
        ]),
    }


@pytest.fixture
def wx_app():
    app = wx.App(False)
    yield app
    app.Destroy()


@pytest.mark.parametrize('a,b,expected', [
    ([.9, .8, .7, .6, .5], [.8, .7, .6, .5, .4], '5. S4 → T: 0.5000'),
    ([.2, 0.0], [.3], 'Only 2 defined transitions available (fewer than 5).'),
    ([], [], 'No defined transitions available.'),
])
def test_gui_summary_full_short_and_empty(a, b, expected):
    text = gui.format_comparative_top_transitions({'A': rankings(a), 'B': rankings(b)})
    assert expected in text
    assert 'positive transitions' not in text.lower()
    if a and 0 in a:
        assert '0.0000' in text


def test_gui_summary_undefined_only():
    # The ranking function must remove undefined values before GUI formatting.
    rows = pd.DataFrame([dict(group='A', source_behavior='A', target_behavior='B',
                              mean_transition_probability=float('nan'), n_animals=1)])
    top = backend.top_group_transitions(rows, ('A', 'B'))
    text = gui.format_comparative_top_transitions({
        'A': top[top.group.eq('A')], 'B': top[top.group.eq('B')]
    })
    assert text.count('No defined transitions available.') == 2


def test_generate_does_not_open_transition_or_statistics_selection(monkeypatch, tmp_path):
    group_a = tmp_path / 'a'
    group_b = tmp_path / 'b'
    output = tmp_path / 'output'
    for folder in (group_a, group_b, output):
        folder.mkdir()
    monkeypatch.setattr(backend, '_workbooks', lambda folder: ['events.xlsx'])
    build = Mock(return_value={
        'top_transitions_by_group': {'A': rankings([.5]), 'B': rankings([0.0])},
    })
    monkeypatch.setattr(backend, 'build_comparative_analysis', build)
    fake = SimpleNamespace(
        group_folders=[str(group_a), str(group_b)], output_folder=str(output),
        behaviors_input=SimpleNamespace(GetValue=lambda: 'A,B'),
        comparison=None, confirmed_pairs=None,
        summary=SimpleNamespace(SetValue=Mock()),
        selection_label=SimpleNamespace(SetLabel=Mock()),
        invalidate_comparison=Mock(), update_actions=Mock(),
        select_transitions=Mock(), run_statistics=Mock(),
    )
    gui.PanelLv2_ComparativeTransitionAnalysis.generate(fake, None)
    build.assert_called_once()
    fake.select_transitions.assert_not_called()
    fake.run_statistics.assert_not_called()
    text = fake.summary.SetValue.call_args.args[0]
    assert 'Comparison generated. Select and confirm transitions' in text
    assert 'TOP 5 TRANSITIONS' not in text


def test_confirm_refreshes_top5_from_confirmed_only(monkeypatch, tmp_path, wx_app):
    frame = wx.Frame(None)
    panel = gui.PanelLv2_ComparativeTransitionAnalysis(frame)
    panel.comparison = comparison_for_confirm_refresh()
    panel.output_folder = str(tmp_path)
    # Seed the matrix dialog selection; Done confirms this set.
    panel.confirmed_pairs = (('b', 'c'), ('c', 'a'))
    panel.statistics_summary = {'stale': True}
    panel.summary.SetValue(
        'STATISTICS\nConfirmed transitions: 99\nSelected metrics: transition_count\n'
    )
    run_statistics = Mock()
    panel.run_statistics = run_statistics
    stats_dialog = Mock()
    monkeypatch.setattr(wx, 'MultiChoiceDialog', stats_dialog)
    monkeypatch.setattr(wx.Dialog, 'ShowModal', lambda self: wx.ID_OK)

    panel.select_transitions(None)

    text = panel.summary.GetValue()
    assert 'GROUP A — TOP 5 TRANSITIONS' in text
    assert 'GROUP B — TOP 5 TRANSITIONS' in text
    assert 'b → c' in text
    assert 'c → a' in text
    assert '0.0000' in text
    assert 'a → b' not in text
    assert 'STATISTICS' not in text
    assert panel.statistics_summary is None
    assert panel.confirmed_pairs == (('b', 'c'), ('c', 'a'))
    run_statistics.assert_not_called()
    stats_dialog.assert_not_called()
    frame.Destroy()
