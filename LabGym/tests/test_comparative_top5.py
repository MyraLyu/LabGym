"""Regression tests for LGDEV-14 group Top 5 transition summaries."""
import numpy as np
import pandas as pd

import LabGym.comparative_transition_analysis as m


def test_top_five_group_independent_order_and_ties():
    rows = []
    for group, probabilities in [('A', [.8, .5, .5, .3, .2, .1]), ('B', [.1, .2, .3, .5, .8, .9])]:
        for index, probability in enumerate(probabilities):
            rows.append(dict(group=group, source_behavior=f'S{index}', target_behavior='T',
                             mean_transition_probability=probability, n_animals=3,
                             n_contributing_transition_probability=3))
    top = m.top_group_transitions(pd.DataFrame(rows), ['A', 'B'])
    assert list(top[top.group.eq('A')].source_behavior) == ['S0', 'S1', 'S2', 'S3', 'S4']
    assert list(top[top.group.eq('B')].source_behavior) == ['S5', 'S4', 'S3', 'S2', 'S1']
    assert list(top[top.group.eq('A')]['rank']) == [1, 2, 3, 4, 5]


def test_undefined_zero_and_empty():
    rows = pd.DataFrame([
        dict(group='A', source_behavior='A', target_behavior='B', mean_transition_probability=np.nan, n_animals=2),
        dict(group='A', source_behavior='B', target_behavior='A', mean_transition_probability=0, n_animals=2),
        dict(group='B', source_behavior='A', target_behavior='B', mean_transition_probability=.25, n_animals=2),
    ])
    top = m.top_group_transitions(rows, ['A', 'B'])
    assert len(top[top.group.eq('A')]) == 1
    assert top[top.group.eq('A')].iloc[0]['mean_transition_probability'] == 0
    assert len(top[top.group.eq('B')]) == 1
    assert top.groupby('group')['rank'].first().eq(1).all()

