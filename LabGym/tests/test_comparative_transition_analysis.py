
"""Regression tests for LabGym comparative transitions: C1 + C2."""

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
# C1: TRANSITION COUNTING
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
# C2: PER-ANIMAL MEASUREMENTS
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


def test_c2_excluded_and_all_na(tmp_path):
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
# C2: FPS PRIORITY AND VALIDATION
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
