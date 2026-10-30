
"""Checkpoint 3 regression tests for comparative transition analysis.

Tests:
- Shared behavior alphabet
- Equal-animal-weighted group summaries
- Observed zero versus undefined probabilities
- Contributor counts
- Aligned transition matrices
- Signed differences
- Interactive transition-selection data
- CSV exports
- Invalid inputs

No inferential statistics.
"""

import numpy as np
import pandas as pd
import pytest

from LabGym.comparative_transition_analysis import (
    build_group_summaries,
    select_transitions,
    plot_comparison_matrices,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def c3_data():
    """Create synthetic C2-style per-animal data.

    Control:
        Animal 0: A -> B probability 1.0
        Animal 1: A -> B probability 0.0
        Animal 2: A -> B undefined

    Treated:
        Animal 0: A -> B probability 0.75
        Animal 1: A -> B probability 0.25

    Expected:
        Control mean = 0.50
        Treated mean = 0.50
        Signed difference = 0.00

    Control contributor count = 2, not 3.
    """

    alphabet = ["A", "B", "C"]

    animals = [
        ("control", "video1", "0", 1.0),
        ("control", "video1", "1", 0.0),
        ("control", "video2", "0", np.nan),
        ("treated", "video3", "0", 0.75),
        ("treated", "video3", "1", 0.25),
    ]

    transition_rows = []
    behavior_rows = []

    for group, recording, animal_id, ab_prob in animals:

        identity = {
            "group": group,
            "source_recording": recording,
            "species": None,
            "animal_id": animal_id,
        }

        if np.isnan(ab_prob):
            outgoing_a = 0
            ab_count = 0
            ac_count = 0
        else:
            outgoing_a = 100
            ab_count = int(round(ab_prob * 100))
            ac_count = 100 - ab_count

        for source in alphabet:
            for target in alphabet:

                if source == target:
                    continue

                if source == "A":
                    outgoing = outgoing_a

                    if target == "B":
                        count = ab_count
                    else:
                        count = ac_count

                else:
                    outgoing = 0
                    count = 0

                probability = (
                    count / outgoing
                    if outgoing > 0
                    else np.nan
                )

                transition_rows.append({
                    **identity,
                    "source_behavior": source,
                    "target_behavior": target,
                    "transition_count": count,
                    "outgoing_count": outgoing,
                    "transition_probability": probability,
                    "probability_defined": outgoing > 0,
                    "switching_rate_per_1000_frames": 10.0,
                })

        for behavior in alphabet:
            behavior_rows.append({
                **identity,
                "behavior": behavior,
                "bout_count": 2,
                "total_bout_frames": 20,
                "mean_bout_frames": 10.0,
                "median_bout_frames": 10.0,
                "mean_bout_duration_seconds": 0.5,
                "median_bout_duration_seconds": 0.5,
                "switching_rate_per_1000_frames": 10.0,
                "valid_included_frames": 100,
                "animal_transition_count": (
                    outgoing_a
                ),
            })

    return (
        pd.DataFrame(transition_rows),
        pd.DataFrame(behavior_rows),
    )


@pytest.fixture
def comparison(c3_data):
    transitions, behaviors = c3_data

    return build_group_summaries(
        transitions,
        behaviors,
        group_order=("control", "treated"),
        behavior_order=["A", "B", "C"],
    )


def get_edge(frame, group, source, target):
    result = frame[
        (frame["group"] == group)
        & (frame["source_behavior"] == source)
        & (frame["target_behavior"] == target)
    ]

    assert len(result) == 1
    return result.iloc[0]


# ============================================================
# TEST 1 — SHARED BEHAVIOR ALPHABET
# ============================================================

def test_shared_behavior_alphabet(comparison):
    assert comparison["behaviors"] == [
        "A", "B", "C"
    ]

    for group in ("control", "treated"):
        matrix = comparison[
            "probability_matrices"
        ][group]

        assert list(matrix.index) == [
            "A", "B", "C"
        ]

        assert list(matrix.columns) == [
            "A", "B", "C"
        ]


def test_custom_behavior_order(c3_data):
    transitions, behaviors = c3_data

    result = build_group_summaries(
        transitions,
        behaviors,
        behavior_order=["C", "A", "B"],
    )

    assert result["behaviors"] == [
        "C", "A", "B"
    ]

    matrix = result["probability_matrices"]["control"]

    assert list(matrix.index) == [
        "C", "A", "B"
    ]


def test_invalid_behavior_order(c3_data):
    transitions, behaviors = c3_data

    with pytest.raises(ValueError):
        build_group_summaries(
            transitions,
            behaviors,
            behavior_order=["A", "B", "D"],
        )


# ============================================================
# TEST 2 — EQUAL ANIMAL WEIGHTING
# ============================================================

def test_equal_animal_weighting(comparison):
    summary = comparison["transition_summary"]

    row = get_edge(
        summary, "control", "A", "B"
    )

    # (1.0 + 0.0) / 2 = 0.5
    # Undefined animal is excluded.
    assert row[
        "mean_transition_probability"
    ] == pytest.approx(0.5)

    # Three animals total.
    assert row["n_animals"] == 3

    # Only two defined probabilities.
    assert row[
        "n_contributing_transition_probability"
    ] == 2


def test_no_pooled_transition_probability(comparison):
    summary = comparison["transition_summary"]

    row = get_edge(
        summary, "control", "A", "B"
    )

    assert row[
        "mean_transition_probability"
    ] == pytest.approx(0.5)


# ============================================================
# TEST 3 — OBSERVED ZERO VS UNDEFINED
# ============================================================

def test_observed_zero_contributes(c3_data):
    transitions, _ = c3_data

    zero = transitions[
        (transitions["group"] == "control")
        & (transitions["animal_id"] == "1")
        & (transitions["source_behavior"] == "A")
        & (transitions["target_behavior"] == "B")
    ].iloc[0]

    assert zero["transition_probability"] == 0
    assert bool(zero["probability_defined"])


def test_undefined_probability_not_zero(comparison):
    summary = comparison["transition_summary"]

    row = get_edge(
        summary, "control", "B", "A"
    )

    assert np.isnan(
        row["mean_transition_probability"]
    )

    assert row[
        "n_contributing_transition_probability"
    ] == 0


def test_undefined_matrix_cell_remains_nan(comparison):
    matrix = comparison[
        "probability_matrices"
    ]["control"]

    assert np.isnan(matrix.loc["B", "A"])


# ============================================================
# TEST 4 — CONTRIBUTOR COUNTS
# ============================================================

def test_group_animal_counts(comparison):
    summary = comparison["transition_summary"]

    control = get_edge(
        summary, "control", "A", "B"
    )

    treated = get_edge(
        summary, "treated", "A", "B"
    )

    assert control["n_animals"] == 3
    assert treated["n_animals"] == 2

    assert control[
        "n_contributing_transition_probability"
    ] == 2

    assert treated[
        "n_contributing_transition_probability"
    ] == 2


def test_behavior_summary_contributors(comparison):
    summary = comparison["behavior_summary"]

    control_a = summary[
        (summary["group"] == "control")
        & (summary["behavior"] == "A")
    ].iloc[0]

    assert control_a["n_animals"] == 3

    assert control_a[
        "n_contributing_mean_bout_duration_seconds"
    ] == 3


# ============================================================
# TEST 5 — SIGNED DIFFERENCES
# ============================================================

def test_signed_difference_zero(comparison):
    diff = comparison["signed_differences"]

    row = diff[
        (diff["source_behavior"] == "A")
        & (diff["target_behavior"] == "B")
    ].iloc[0]

    assert row[
        "difference_transition_probability"
    ] == pytest.approx(0.0)


def test_signed_difference_positive(c3_data):
    transitions, behaviors = c3_data

    # Change treated animals to probability 1.
    mask = (
        (transitions["group"] == "treated")
        & (transitions["source_behavior"] == "A")
        & (transitions["target_behavior"] == "B")
    )

    transitions.loc[
        mask, "transition_probability"
    ] = 1.0

    result = build_group_summaries(
        transitions,
        behaviors,
        group_order=("control", "treated"),
    )

    diff = result["signed_differences"]

    row = diff[
        (diff["source_behavior"] == "A")
        & (diff["target_behavior"] == "B")
    ].iloc[0]

    assert row[
        "difference_transition_probability"
    ] == pytest.approx(0.5)


def test_undefined_difference_remains_nan(comparison):
    matrix = comparison["difference_matrix"]

    assert np.isnan(matrix.loc["B", "A"])


# ============================================================
# TEST 6 — MATRIX ALIGNMENT
# ============================================================

def test_aligned_matrices(comparison):
    control = comparison[
        "probability_matrices"
    ]["control"]

    treated = comparison[
        "probability_matrices"
    ]["treated"]

    diff = comparison["difference_matrix"]

    assert control.shape == (3, 3)
    assert treated.shape == (3, 3)
    assert diff.shape == (3, 3)

    assert control.index.equals(treated.index)
    assert control.columns.equals(treated.columns)

    assert diff.index.equals(control.index)
    assert diff.columns.equals(control.columns)


def test_matrix_difference_matches_summary(comparison):
    control = comparison[
        "probability_matrices"
    ]["control"]

    treated = comparison[
        "probability_matrices"
    ]["treated"]

    diff = comparison["difference_matrix"]

    expected = (
        treated.loc["A", "B"]
        - control.loc["A", "B"]
    )

    assert diff.loc["A", "B"] == pytest.approx(expected)


# ============================================================
# TEST 7 — TRANSITION SELECTION
# ============================================================

def test_select_single_transition(comparison):
    summary, diff = select_transitions(
        comparison,
        [("A", "B")],
    )

    assert len(summary) == 2
    assert len(diff) == 1

    assert set(summary["source_behavior"]) == {"A"}
    assert set(summary["target_behavior"]) == {"B"}


def test_select_multiple_transitions(comparison):
    summary, diff = select_transitions(
        comparison,
        [("A", "B"), ("B", "C")],
    )

    assert len(summary) == 4
    assert len(diff) == 2


def test_invalid_transition_selection(comparison):
    with pytest.raises(ValueError):
        select_transitions(
            comparison,
            [("A", "D")],
        )


def test_empty_selection(comparison):
    summary, diff = select_transitions(
        comparison,
        [],
    )

    assert summary.empty
    assert diff.empty


# ============================================================
# TEST 8 — EXPORTS
# ============================================================

def test_group_csv_exports(c3_data, tmp_path):
    transitions, behaviors = c3_data

    result = build_group_summaries(
        transitions,
        behaviors,
        group_order=("control", "treated"),
        output_folder=tmp_path,
    )

    expected_files = [
        "group_transition_summary.csv",
        "group_behavior_summary.csv",
        "group_signed_differences.csv",
        "group_1_probability_matrix.csv",
        "group_2_probability_matrix.csv",
        "signed_difference_matrix.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).is_file()

    exported = pd.read_csv(
        tmp_path / "group_transition_summary.csv"
    )

    assert "n_animals" in exported.columns

    assert (
        "n_contributing_transition_probability"
        in exported.columns
    )


def test_export_preserves_nan(c3_data, tmp_path):
    transitions, behaviors = c3_data

    build_group_summaries(
        transitions,
        behaviors,
        output_folder=tmp_path,
    )

    matrix = pd.read_csv(
        tmp_path / "group_1_probability_matrix.csv",
        index_col=0,
    )

    assert np.isnan(matrix.loc["B", "A"])


# ============================================================
# TEST 9 — INPUT VALIDATION
# ============================================================

def test_duplicate_animal_behavior_rejected(c3_data):
    transitions, behaviors = c3_data

    duplicate = behaviors.iloc[[0]].copy()

    behaviors = pd.concat(
        [behaviors, duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_group_summaries(
            transitions,
            behaviors,
        )


def test_missing_group_rejected(c3_data):
    transitions, behaviors = c3_data

    transitions = transitions[
        transitions["group"] == "control"
    ]

    behaviors = behaviors[
        behaviors["group"] == "control"
    ]

    with pytest.raises(ValueError):
        build_group_summaries(
            transitions,
            behaviors,
        )


def test_mismatched_identity_rejected(c3_data):
    transitions, behaviors = c3_data

    behaviors = behaviors.copy()

    behaviors.loc[
        behaviors["animal_id"] == "1",
        "animal_id",
    ] = "changed"

    with pytest.raises(ValueError):
        build_group_summaries(
            transitions,
            behaviors,
        )


def test_incompatible_behavior_alphabet(c3_data):
    transitions, behaviors = c3_data

    behaviors = behaviors[
        ~(
            (behaviors["group"] == "treated")
            & (behaviors["behavior"] == "C")
        )
    ]

    with pytest.raises(ValueError):
        build_group_summaries(
            transitions,
            behaviors,
        )


# ============================================================
# TEST 10 — VISUALIZATION
# ============================================================

def test_plot_comparison_matrices(comparison, tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output = tmp_path / "comparison.png"

    fig = plot_comparison_matrices(
        comparison,
        output_path=output,
        selected_pairs=[("A", "B")],
    )

    assert output.is_file()
    assert len(fig.axes) >= 3

    plt.close(fig)
