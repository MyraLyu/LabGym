"""Regression tests for LabGym comparative transition analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LabGym.comparative_transition_analysis import (
    _fps,
    load_recording,
    build_transition_dataset,
    build_per_animal_measurements,
)


# ============================================================
# TEST HELPERS
# ============================================================

def write_result_workbook(
    parent,
    folder_name,
    filename,
    animals,
    times=None,
):
    folder = Path(parent) / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    sequences = list(animals.values())
    nrow = len(sequences[0])

    if any(len(seq) != nrow for seq in sequences):
        raise ValueError("Unequal animal sequence lengths")

    if times is None:
        times = list(range(nrow))

    data = {"time/ID": list(times)}

    for animal_id, cells in animals.items():
        data[str(animal_id)] = [
            str([cell, 0.9])
            if not isinstance(cell, str) or not cell.startswith("[")
            else cell
            for cell in cells
        ]

    pd.DataFrame(data).to_excel(
        folder / filename,
        index=False,
    )

    return folder


def configs(a, b, fps=None, excluded=()):
    result = [
        {
            "folder": a,
            "group": "control",
            "behaviors": ["A", "B", "C"],
            "excluded": list(excluded),
        },
        {
            "folder": b,
            "group": "treated",
            "behaviors": ["A", "B", "C"],
            "excluded": list(excluded),
        },
    ]

    if fps is not None:
        for rec in result:
            rec["fps"] = fps

    return result


def clock(fps, n, start_frame=0):
    """LabGym-style centisecond-rounded frame timestamps."""
    return np.round(
        (np.arange(n) + start_frame) / fps,
        2,
    )


def labgym_time_index(fps, n, t=0, length=15):
    """Timestamps from LabGym's centisecond time/ID writer."""
    start_t = round((t - length / fps), 2)
    if start_t < 0:
        start_t = 0.00

    first_frame = 1
    while round(first_frame / fps, 2) < start_t:
        first_frame += 1

    return [
        round(
            round((first_frame + i) / fps, 2) - start_t,
            2,
        )
        for i in range(n)
    ]


def make_pair(
    tmp_path,
    first,
    second,
    fps=10,
    n=None,
):
    if n is None:
        n = max(len(first), len(second))

    a = write_result_workbook(
        tmp_path,
        "video_a",
        "all_event_probability.xlsx",
        {"0": first},
        clock(fps, len(first)),
    )

    b = write_result_workbook(
        tmp_path,
        "video_b",
        "all_event_probability.xlsx",
        {"0": second},
        clock(fps, len(second)),
    )

    return a, b


# ============================================================
# TRANSITION COUNTING
# ============================================================

def test_na_break(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "na",
        "all_event_probability.xlsx",
        {"0": ["A", "NA", "B"]},
    )

    result = load_recording(
        folder, "control", ["A", "B"]
    )

    assert result["transition_count"].sum() == 0


def test_excluded_behavior_break(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "excluded",
        "all_event_probability.xlsx",
        {"0": ["A", "C", "B"]},
    )

    result = load_recording(
        folder,
        "control",
        ["A", "B", "C"],
        excluded=["C"],
    )

    assert result["transition_count"].sum() == 0


def test_absent_behavior_undefined(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "absent",
        "all_event_probability.xlsx",
        {"0": ["A", "B"]},
    )

    result = load_recording(
        folder, "control", ["A", "B", "C"]
    )

    absent = result[
        result["source_behavior"] == "C"
    ]

    assert (absent["transition_count"] == 0).all()
    assert (absent["outgoing_count"] == 0).all()
    assert absent["transition_probability"].isna().all()
    assert not absent["probability_defined"].any()


def test_observed_zero_is_defined(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "zero",
        "all_event_probability.xlsx",
        {"0": ["A", "B"]},
    )

    result = load_recording(
        folder, "control", ["A", "B", "C"]
    )

    ab = result[
        (result["source_behavior"] == "A")
        & (result["target_behavior"] == "B")
    ].iloc[0]

    ac = result[
        (result["source_behavior"] == "A")
        & (result["target_behavior"] == "C")
    ].iloc[0]

    assert ab["transition_probability"] == 1
    assert ac["transition_probability"] == 0
    assert bool(ac["probability_defined"])


def test_single_bout(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "single",
        "all_event_probability.xlsx",
        {"0": ["A", "A", "A"]},
    )

    result = load_recording(
        folder, "control", ["A", "B", "C"]
    )

    assert result["transition_count"].sum() == 0


def test_incompatible_vocab(tmp_path):
    a, b = make_pair(
        tmp_path,
        ["A", "B"],
        ["B", "A"],
    )

    with pytest.raises(ValueError, match="Incompatible"):
        build_transition_dataset([
            {
                "folder": a,
                "group": "control",
                "behaviors": ["A", "B", "C"],
            },
            {
                "folder": b,
                "group": "treated",
                "behaviors": ["A", "B", "D"],
            },
        ])


def test_background_workbook(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "background",
        "all_event_probability.xlsx",
        {"0": ["A", "B"]},
    )

    result = load_recording(
        folder, "control", ["A", "B"]
    )

    assert result["transition_count"].sum() == 1
    assert result["species"].isna().all()


def test_detector_species(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "detector",
        "mouse_all_event_probability.xlsx",
        {"0": ["A", "B"]},
    )

    write_result_workbook(
        tmp_path,
        "detector",
        "fly_all_event_probability.xlsx",
        {"0": ["B", "A"]},
    )

    result = load_recording(
        folder, "control", ["A", "B"]
    )

    assert set(result["species"]) == {
        "mouse", "fly"
    }
    assert result["transition_count"].sum() == 2


def test_batch_workbook_ignored(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "valid",
        "all_event_probability.xlsx",
        {"0": ["A", "B"]},
    )

    (folder / "all_events.xlsx").write_bytes(
        b"invalid workbook"
    )

    result = load_recording(
        folder, "control", ["A", "B"]
    )

    assert result["transition_count"].sum() == 1

    batch_only = tmp_path / "batch_only"
    batch_only.mkdir()

    (batch_only / "all_events.xlsx").write_bytes(
        b"invalid workbook"
    )

    with pytest.raises(FileNotFoundError):
        load_recording(
            batch_only, "control", ["A", "B"]
        )


def test_same_animal_id_separate_recordings(tmp_path):
    a, b = make_pair(
        tmp_path,
        ["A", "B"],
        ["B", "A"],
    )

    result = build_transition_dataset(
        configs(a, b)
    )

    assert result["source_recording"].nunique() == 2
    assert set(result["animal_id"]) == {"0"}
    assert result["transition_count"].sum() == 2


def test_all_na_animal_retained(tmp_path):
    folder = write_result_workbook(
        tmp_path,
        "all_na",
        "all_event_probability.xlsx",
        {"0": ["NA", "NA", "NA"]},
    )

    result = load_recording(
        folder, "control", ["A", "B"]
    )

    assert not result.empty
    assert result["transition_count"].sum() == 0
    assert result["transition_probability"].isna().all()


# ============================================================
# PER-ANIMAL MEASUREMENTS
# ============================================================

def test_per_animal_bouts(tmp_path):
    a, b = make_pair(
        tmp_path,
        ["A", "A", "B", "B", "NA", "A", "C"],
        ["B", "B", "B", "B", "B", "B", "B"],
    )

    t, s, bouts = build_per_animal_measurements(
        configs(a, b, fps=10)
    )

    ta = t[
        t["source_recording"] == str(a.resolve())
    ]

    assert ta["transition_count"].sum() == 2
    assert ta["valid_included_frames"].iloc[0] == 6
    assert ta[
        "switching_rate_per_1000_frames"
    ].iloc[0] == pytest.approx(2000 / 6)

    aa = s[
        (s["source_recording"] == str(a.resolve()))
        & (s["behavior"] == "A")
    ].iloc[0]

    assert aa["bout_count"] == 2
    assert aa["total_bout_frames"] == 3
    assert aa[
        "mean_bout_duration_seconds"
    ] == pytest.approx(0.15)


def test_no_animal_pooling(tmp_path):
    a = write_result_workbook(
        tmp_path,
        "a",
        "all_event_probability.xlsx",
        {
            "0": ["A", "B"],
            "1": ["A", "A"],
        },
    )

    b = write_result_workbook(
        tmp_path,
        "b",
        "all_event_probability.xlsx",
        {"0": ["B", "A"]},
    )

    t, s, _ = build_per_animal_measurements(
        configs(a, b, fps=10)
    )

    identities = s[
        ["source_recording", "animal_id"]
    ].drop_duplicates()

    assert len(identities) == 3
    assert t["transition_count"].sum() == 2


def test_excluded_and_all_na(tmp_path):
    a = write_result_workbook(
        tmp_path,
        "a",
        "all_event_probability.xlsx",
        {
            "0": ["A", "C", "B"],
            "1": ["NA", "NA", "NA"],
        },
    )

    b = write_result_workbook(
        tmp_path,
        "b",
        "all_event_probability.xlsx",
        {"0": ["A", "B", "B"]},
    )

    t, s, _ = build_per_animal_measurements(
        configs(a, b, fps=10, excluded=["C"])
    )

    first = t[
        t["source_recording"] == str(a.resolve())
    ]

    assert first["transition_count"].sum() == 0

    all_na = s[
        (s["source_recording"] == str(a.resolve()))
        & (s["animal_id"] == "1")
    ]

    assert all_na["valid_included_frames"].eq(0).all()
    assert all_na[
        "switching_rate_per_1000_frames"
    ].isna().all()


def test_mixed_fps_export(tmp_path):
    a = write_result_workbook(
        tmp_path,
        "a",
        "all_event_probability.xlsx",
        {"0": ["A", "A", "B", "B"]},
        clock(10, 4),
    )

    b = write_result_workbook(
        tmp_path,
        "b",
        "all_event_probability.xlsx",
        {"0": ["B", "B", "A", "A"]},
        clock(20, 4),
    )

    records = configs(a, b)
    records[0]["fps"] = 10
    records[1]["fps"] = 20

    out = tmp_path / "export"

    t, s, bouts = build_per_animal_measurements(
        records,
        output_folder=out,
    )

    for frame in (t, s, bouts):
        assert set(frame["fps"]) == {10.0, 20.0}
        assert set(frame["fps_source"]) == {"explicit"}
        assert frame["source_recording"].nunique() == 2

    for name in (
        "per_animal_transitions.csv",
        "per_animal_behaviors.csv",
        "per_animal_bouts.csv",
    ):
        exported = pd.read_csv(out / name)
        assert set(exported["fps"]) == {10, 20}


def test_species_export(tmp_path):
    a = write_result_workbook(
        tmp_path,
        "a",
        "mouse_all_event_probability.xlsx",
        {"0": ["A", "B"]},
    )

    b = write_result_workbook(
        tmp_path,
        "b",
        "fly_all_event_probability.xlsx",
        {"0": ["B", "A"]},
    )

    t, s, bouts = build_per_animal_measurements(
        configs(a, b, fps=10)
    )

    for frame in (t, s, bouts):
        assert set(frame["species"]) == {
            "mouse", "fly"
        }


# ============================================================
# FPS PRIORITY AND VALIDATION
# ============================================================

def test_explicit_fps_priority(tmp_path):
    fps, source = _fps(
        tmp_path,
        [0.00, 0.03, 0.07],
        explicit=30,
    )

    assert fps == 30
    assert source == "explicit"


def test_analysis_log_fps_priority(tmp_path):
    (tmp_path / "Analysis log.txt").write_text(
        "Video fps: 25\n"
    )

    fps, source = _fps(
        tmp_path,
        [0.00, 0.04, 0.08],
    )

    assert fps == 25
    assert source == "Analysis log.txt"


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_ambiguous_short_index(tmp_path, length):
    times = clock(30, length)

    with pytest.raises(
        ValueError, match="Ambiguous short"
    ):
        _fps(tmp_path, times)


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_ambiguous_short_offset_index(tmp_path, length):
    times = clock(
        30, length, start_frame=1
    )

    with pytest.raises(
        ValueError, match="Ambiguous short"
    ):
        _fps(tmp_path, times)


def test_default_30fps_clock(tmp_path):
    times = clock(30, 120)

    fps, source = _fps(tmp_path, times)

    assert fps == 30
    assert source == "time/ID (centisecond-rounded)"


def test_default_30fps_seven_timestamps(tmp_path):
    """Single-round 0.00, 0.03, ... matches more than one fps."""
    times = clock(30, 7)

    with pytest.raises(
        ValueError, match="Ambiguous time/ID fps"
    ):
        _fps(tmp_path, times)


@pytest.mark.parametrize(
    "t, length",
    [
        (0, 15),
        (0.53, 15),
    ],
)
def test_offset_start_30fps(tmp_path, t, length):
    times = labgym_time_index(
        30, 120, t=t, length=length
    )

    fps, source = _fps(tmp_path, times)

    assert fps == 30
    assert source == "time/ID (centisecond-rounded)"


def test_nonzero_analysis_start_30fps(tmp_path):
    times = labgym_time_index(
        30, 120, t=0.53, length=15
    )

    assert times[:5] == [0.00, 0.04, 0.07, 0.10, 0.14]

    fps, source = _fps(tmp_path, times)

    assert fps == 30
    assert source == "time/ID (centisecond-rounded)"


def test_nonzero_analysis_start_30fps_seven_timestamps(
    tmp_path,
):
    times = labgym_time_index(
        30, 7, t=0.53, length=15
    )

    fps, source = _fps(tmp_path, times)

    assert fps == 30
    assert source == "time/ID (centisecond-rounded)"


def test_labgym_clamped_start_clock(tmp_path):
    times = np.array([
        0.03, 0.07, 0.10, 0.13,
        0.17, 0.20, 0.23,
    ])

    fps, source = _fps(tmp_path, times)

    assert fps == 30
    assert source == "time/ID (centisecond-rounded)"


@pytest.mark.parametrize(
    "times",
    [
        [0.00, 0.03, 0.03, 0.10, 0.13, 0.17, 0.20],
        [0.00, 0.03, 0.02, 0.10, 0.13, 0.17, 0.20],
        [0, 1, 2, 3, 4, 5, 6],
        [0.00, 0.03, 0.07, 0.50, 0.53, 0.57, 0.60],
    ],
)
def test_invalid_time_indexes(tmp_path, times):
    with pytest.raises(ValueError):
        _fps(tmp_path, times)


def test_rounded_clock_end_to_end(tmp_path):
    a = write_result_workbook(
        tmp_path,
        "a",
        "all_event_probability.xlsx",
        {"0": ["A"] * 120},
        labgym_time_index(30, 120, t=0, length=15),
    )

    b = write_result_workbook(
        tmp_path,
        "b",
        "all_event_probability.xlsx",
        {"0": ["B"] * 120},
        labgym_time_index(30, 120, t=0.53, length=15),
    )

    t, s, bouts = build_per_animal_measurements(
        configs(a, b)
    )

    for frame in (t, s, bouts):
        assert set(frame["fps"]) == {30.0}
        assert set(frame["fps_source"]) == {
            "time/ID (centisecond-rounded)"
        }

    assert s[
        "mean_bout_duration_seconds"
    ].dropna().eq(4.0).all()

"""Two-group statistical regression tests."""
import LabGym.comparative_transition_analysis as m


def sample(a=(1,)*5, b=(0,)*5, fps_a=30, fps_b=30):
    rows = []
    for group, values, fps in [('Control', a, fps_a), ('Treatment', b, fps_b)]:
        for i, value in enumerate(values):
            rows.append(dict(group=group, source_recording=f'{group}_{i}', species='mouse',
                             animal_id='0', source_behavior='walk', target_behavior='rest',
                             transition_count=value, outgoing_count=10,
                             transition_probability=value,
                             probability_defined=np.isfinite(value),
                             switching_rate_per_1000_frames=value, fps=fps, fps_source='explicit'))
    transitions = pd.DataFrame(rows)
    behaviors = pd.DataFrame([{**r, 'behavior':'walk', 'bout_count':r['transition_count'],
                               'total_bout_frames':r['transition_count'],
                               'mean_bout_frames':r['transition_count'],
                               'median_bout_frames':r['transition_count'],
                               'mean_bout_duration_seconds':r['transition_count'],
                               'median_bout_duration_seconds':r['transition_count']} for r in rows])
    return dict(groups=['Control','Treatment'], per_animal_transitions=transitions,
                per_animal_behaviors=behaviors)


def test_known_effect_and_cliffs_delta():
    row = m.build_two_group_statistics(sample(), [('walk','rest')])['results'].iloc[0]
    assert row.testable and row.test_type == 'exact'
    assert row.permutation_count == 252
    assert row.mean_difference_a_minus_b == 1
    assert row.cliffs_delta_a_vs_b == 1
    assert row.p_raw == pytest.approx(2/252)
    assert row.significant_holm
    assert row.difference_direction == 'Control - Treatment'


def test_no_effect_is_reported_not_discarded():
    row = m.build_two_group_statistics(sample(b=(1,)*5), [('walk','rest')])['results'].iloc[0]
    assert row.testable and row.p_raw == 1
    assert row.p_holm == 1 and not row.significant_holm


def test_low_support_is_exported_with_reason(tmp_path):
    result = m.build_two_group_statistics(sample(a=(0,0,0,0,np.nan)), [('walk','rest')], output_folder=tmp_path)
    row = result['results'].iloc[0]
    assert not row.testable and row.n_contributing_a == 4 and row.n_excluded_a == 1
    assert 'insufficient_animals' in row.non_testable_reason
    assert np.isnan(row.p_raw) and np.isnan(row.p_holm)
    assert len(pd.read_csv(tmp_path/'two_group_permutation_statistics.csv')) == 1


def test_observed_zero_is_valid_and_undefined_is_excluded():
    c = sample(a=(0,0,0,0,0,np.nan), b=(0,)*5)
    result = m.build_two_group_statistics(c, [('walk','rest')])
    row = result['results'].iloc[0]
    assert row.testable and row.n_contributing_a == 5 and row.n_excluded_a == 1
    assert row.mean_a == 0 and row.p_raw == 1
    animals = result['per_animal_values']
    assert animals.excluded_from_test.sum() == 1
    assert (animals.loc[animals.measurement_value.eq(0), 'excluded_from_test'] == False).all()


def test_fps_incompatibility_blocks_only_switching_rate():
    c = sample(fps_b=60)
    specs = [dict(kind='transition', source_behavior='walk',target_behavior='rest',
                  metric='switching_rate_per_1000_frames'), ('walk','rest'),
             dict(kind='transition',source_behavior='walk',target_behavior='rest',metric='transition_count')]
    rows = m.build_two_group_statistics(c, specs)['results']
    assert not rows.iloc[0].testable and 'incompatible_fps' in rows.iloc[0].non_testable_reason
    assert rows.iloc[1].testable and rows.iloc[2].testable
    assert 'raw_counts_not_exposure_normalized' in rows.iloc[2].warnings


def test_exact_threshold_and_randomized_seed():
    exact = m.build_two_group_statistics(sample(a=(1,)*9,b=(0,)*9), [('walk','rest')])['results'].iloc[0]
    assert exact.test_type == 'exact' and exact.permutation_count == 48620
    c = sample(a=(1,)*10,b=(0,)*10)
    one = m.build_two_group_statistics(c, [('walk','rest')], seed=123)['results'].iloc[0]
    two = m.build_two_group_statistics(c, [('walk','rest')], seed=123)['results'].iloc[0]
    assert one.test_type == 'randomized' and one.permutation_count == 100000
    assert one.seed == 123 and one.p_raw == two.p_raw
    assert 0 < one.p_raw <= 1


def test_holm_all_eligible_only():
    c = sample()
    specs = [('walk','rest'),
             dict(kind='behavior',behavior='walk',metric='mean_bout_duration_seconds'),
             dict(kind='transition',source_behavior='walk',target_behavior='rest',metric='switching_rate_per_1000_frames')]
    c['per_animal_transitions'].loc[c['per_animal_transitions'].group.eq('Treatment'), 'fps'] = 60
    rows = m.build_two_group_statistics(c, specs)['results']
    assert rows.testable.tolist() == [True, True, False]
    assert rows.loc[:1,'p_holm'].tolist() == pytest.approx([4/252, 4/252])
    assert np.isnan(rows.iloc[2].p_holm)


def test_all_eligible_measurement_families_and_traceability(tmp_path):
    c = sample()
    specs = [dict(kind='transition',source_behavior='walk',target_behavior='rest',metric=x)
             for x in ['transition_count','transition_probability','switching_rate_per_1000_frames']]
    specs += [dict(kind='behavior', behavior='walk', metric=x)
              for x in ['mean_bout_duration_seconds','median_bout_duration_seconds',]]
    output = m.build_two_group_statistics(c,specs,output_folder=tmp_path)
    assert len(output['results']) == len(specs)
    assert len(output['per_animal_values']) == len(specs)*10
    assert {'group','source_recording','species','animal_id','fps','measurement_value',
            'selection_label','excluded_from_test'}.issubset(output['per_animal_values'].columns)
    exported = pd.read_csv(tmp_path/'two_group_permutation_statistics.csv')
    assert len(exported) == len(specs)
    assert {'n_contributing_a','n_contributing_b','n_excluded_a','n_excluded_b',
            'mean_difference_a_minus_b','cliffs_delta_a_vs_b','p_raw','p_holm',
            'warnings','test_type','permutation_count','seed','non_testable_reason'}.issubset(exported.columns)


def test_only_confirmed_and_duplicate_guard():
    c = sample()
    assert m.build_two_group_statistics(c,[])['results'].empty
    with pytest.raises(ValueError,match='Duplicate confirmed'):
        m.build_two_group_statistics(c,[('walk','rest'),('walk','rest')])


def test_reverse_display_does_not_reverse_statistical_contract():
    c = sample()
    c['difference_direction'] = 'B-A'
    row = m.build_two_group_statistics(c,[('walk','rest')])['results'].iloc[0]
    assert row.mean_difference_a_minus_b == 1
    assert row.difference_direction == 'Control - Treatment'



def fixture(a, b, metric='transition_count', fps_a=30, fps_b=30):
    rows = []
    for group, values, fps in [('A', a, fps_a), ('B', b, fps_b)]:
        for i, value in enumerate(values):
            rows.append(dict(group=group, source_recording=f'{group}_video', species=None,
                             animal_id=str(i), source_behavior='walk', target_behavior='rest',
                             transition_count=value, transition_probability=value,
                             switching_rate_per_1000_frames=value, fps=fps))
    transitions = pd.DataFrame(rows)
    behaviors = transitions.rename(columns={'source_behavior':'behavior'}).copy()
    behaviors['mean_bout_duration_seconds'] = behaviors['transition_count']
    behaviors['median_bout_duration_seconds'] = behaviors['transition_count']
    return {'groups':['A','B'], 'per_animal_transitions':transitions,
            'per_animal_behaviors':behaviors}


def test_negative_effect_and_cliffs_delta():
    result = m.build_two_group_statistics(fixture([1]*5,[5]*5), [('walk','rest')])['results'].iloc[0]
    assert result.mean_difference_a_minus_b == -4
    assert result.cliffs_delta_a_vs_b == -1
    assert result.test_type == 'exact'
    assert result.permutation_count == 252


def test_non_testable_significance_null(tmp_path):
    result = m.build_two_group_statistics(fixture([1]*4,[2]*5), [('walk','rest')], output_folder=tmp_path)['results'].iloc[0]
    assert result.testable == False
    assert pd.isna(result.significant_holm)
    assert pd.isna(result.p_raw) and pd.isna(result.p_holm)
    exported = pd.read_csv(tmp_path/'two_group_permutation_statistics.csv')
    assert pd.isna(exported.loc[0,'significant_holm'])


def test_disallowed_metrics():
    for metric in ['bout_count','total_bout_frames','mean_bout_frames','median_bout_frames']:
        with pytest.raises(ValueError, match='Unsupported'):
            m._normalize_confirmed_selection({'kind':'behavior','behavior':'walk','metric':metric})


def test_randomized_seed_reproducible():
    a=np.arange(11,dtype=float)
    b=np.arange(11,22,dtype=float)
    r1=m._permutation_test(a,b,seed=17)
    r2=m._permutation_test(a,b,seed=17)
    assert r1==r2
    assert r1[1:] == ('randomized',100000,17)


def test_holm_and_no_effect():
    comparison=fixture([2]*5,[2]*5)
    result=m.build_two_group_statistics(comparison,[('walk','rest'),
        {'kind':'behavior','behavior':'walk','metric':'mean_bout_duration_seconds'}])['results']
    assert len(result)==2
    assert (result['p_raw']==1).all()
    assert (result['p_holm']==1).all()
    assert (result['significant_holm']==False).all()


def test_undefined_probability_excluded_zero_kept():
    comparison=fixture([0,0,0,0,np.nan],[1]*5)
    result=m.build_two_group_statistics(comparison,[('walk','rest')])['results'].iloc[0]
    # Selection shorthand uses transition probability: one undefined is excluded.
    assert result.n_contributing_a == 4
    assert result.n_excluded_a == 1
    assert not result.testable


def test_fps_block_only_switching_rate():
    comparison=fixture([1]*5,[2]*5,fps_a=30,fps_b=60)
    selections=[{'kind':'transition','source_behavior':'walk','target_behavior':'rest',
                 'metric':metric} for metric in ['transition_count','switching_rate_per_1000_frames']]
    result=m.build_two_group_statistics(comparison,selections)['results'].set_index('metric')
    assert bool(result.loc['transition_count','testable'])
    assert not bool(result.loc['switching_rate_per_1000_frames','testable'])
