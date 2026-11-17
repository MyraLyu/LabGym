"""GUI-facing Top 5 regression tests (requires wxPython / LabGym GUI dependencies)."""
from pathlib import Path
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
    assert 'GROUP A' in text and 'GROUP B' in text
    assert '0.0000' in text
