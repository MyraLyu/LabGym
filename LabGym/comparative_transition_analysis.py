"""LabGym comparative transition analysis: per-animal measurements, group comparisons and permutation inference."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = [
    "group",
    "source_recording",
    "species",
    "animal_id",
]


# ============================================================
# — INPUTS AND TRANSITION COUNTS
# ============================================================

def _event(value):
    """Extract a behavior label from a LabGym event cell."""
    if isinstance(value, (list, tuple)):
        item = value
    elif pd.isna(value):
        return None
    else:
        try:
            item = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            raise ValueError(
                f"Invalid event cell: {value!r}"
            ) from None

    if not isinstance(item, (list, tuple)) or not item:
        raise ValueError(f"Invalid event: {value!r}")

    state = item[0]

    if (
        state is None
        or str(state).strip().upper()
        in ("NA", "NAN", "NONE", "")
    ):
        return None

    return str(state)


def _workbooks(folder):
    """Discover only per-video event-probability workbooks."""
    folder = Path(folder)
    suffix = "_all_event_probability.xlsx"
    background = folder / "all_event_probability.xlsx"

    paths = []

    if background.is_file():
        paths.append((background, None))

    for path in sorted(folder.glob("*" + suffix)):
        if path.name == background.name:
            continue

        species = path.name[:-len(suffix)]

        if species:
            paths.append((path, species))

    return paths


def load_recording(
    folder,
    group,
    behaviors,
    excluded=(),
    collapse_repeats=True,
    include_self=False,
):
    """Load one analyzed video without merging animal identities."""
    folder = Path(folder)
    paths = _workbooks(folder)

    if not paths:
        raise FileNotFoundError(
            "No per-video event probability workbook "
            f"found in {folder}"
        )

    frames = []

    for path, species in paths:
        df = pd.read_excel(path)

        frame = _load_workbook(
            df,
            path,
            folder,
            group,
            species,
            behaviors,
            excluded,
            collapse_repeats,
            include_self,
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _load_workbook(
    df,
    path,
    folder,
    group,
    species,
    behaviors,
    excluded,
    collapse_repeats,
    include_self,
):
    if "time/ID" not in df.columns:
        raise ValueError(f"Missing time/ID in {path}")

    if not df["time/ID"].is_unique:
        raise ValueError(f"Duplicate timestamps in {path}")

    vocab = list(behaviors)

    if not vocab or len(vocab) != len(set(vocab)):
        raise ValueError(
            "Behavior vocabulary must be nonempty and unique"
        )

    excluded = set(excluded)

    if not excluded.issubset(vocab):
        raise ValueError(
            f"Unknown excluded behaviors: {excluded - set(vocab)}"
        )

    selected = [
        behavior for behavior in vocab
        if behavior not in excluded
    ]

    if not selected:
        raise ValueError(
            "No behaviors remain after exclusions"
        )

    rows = []

    for col in df.columns:
        if col == "time/ID":
            continue

        animal_id = str(col)
        sequence = []
        bouts = []

        def flush():
            if sequence:
                bouts.append(sequence.copy())
                sequence.clear()

        for value in df[col]:
            state = _event(value)

            if state is None or state in excluded:
                flush()
                continue

            if state not in vocab:
                raise ValueError(
                    f"Unknown behavior {state!r} "
                    f"in {path}, animal {animal_id}"
                )

            if (
                not collapse_repeats
                or not sequence
                or sequence[-1] != state
            ):
                sequence.append(state)

        flush()

        counts = {
            (a, b): 0
            for a in selected
            for b in selected
            if include_self or a != b
        }

        for bout in bouts:
            for a, b in zip(bout, bout[1:]):
                if (a, b) in counts:
                    counts[(a, b)] += 1

        for a in selected:
            denominator = sum(
                count
                for (source, _), count in counts.items()
                if source == a
            )

            for b in selected:
                if not include_self and a == b:
                    continue

                count = counts[(a, b)]

                rows.append({
                    "group": str(group),
                    "source_recording": str(folder.resolve()),
                    "species": species,
                    "animal_id": animal_id,
                    "source_behavior": a,
                    "target_behavior": b,
                    "transition_count": count,
                    "outgoing_count": denominator,
                    "transition_probability": (
                        count / denominator
                        if denominator else np.nan
                    ),
                    "probability_defined": bool(denominator),
                })

    return pd.DataFrame(rows)


def build_transition_dataset(
    recordings,
    output_csv=None,
    collapse_repeats=True,
    include_self=False,
):
    """Build a two-group transition dataset without animal pooling."""
    recordings = list(recordings)

    if not recordings:
        raise ValueError("No recordings supplied")

    groups = {
        str(recording["group"])
        for recording in recordings
    }

    if len(groups) != 2:
        raise ValueError(
            "First version requires exactly two groups"
        )

    expected = set(recordings[0]["behaviors"])
    excluded = set(recordings[0].get("excluded", ()))

    frames = []
    identities = set()

    for recording in recordings:
        if set(recording["behaviors"]) != expected:
            raise ValueError(
                "Incompatible behavior sets: "
                f"{recording['folder']}"
            )

        if set(recording.get("excluded", ())) != excluded:
            raise ValueError(
                "All recordings must use the same exclusions"
            )

        folder = Path(recording["folder"]).resolve()

        identity = (
            str(recording["group"]),
            str(folder),
        )

        if identity in identities:
            raise ValueError(
                f"Duplicate recording: {folder}"
            )

        identities.add(identity)

        frame = load_recording(
            folder,
            recording["group"],
            recording["behaviors"],
            excluded,
            collapse_repeats,
            include_self,
        )

        frame["source_recording"] = str(folder)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)

    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(
            parents=True, exist_ok=True
        )
        result.to_csv(destination, index=False)

    return result


# ============================================================
# — FPS INFERENCE
# ============================================================

def _fps(folder, time_index, explicit=None):
    """Resolve FPS from explicit input, analysis log, or frame clock.

    The time/ID fallback reproduces LabGym's centisecond-rounded
    timestamp writer, including nonzero analysis-start offsets.

    Multiple matching integer FPS candidates are rejected.
    """
    folder = Path(folder)

    if explicit is not None:
        fps = float(explicit)
        source = "explicit"

    else:
        fps = None
        source = None

        log = folder / "Analysis log.txt"

        if log.is_file():
            match = re.search(
                r"Video\s*fps\s*[:=]\s*"
                r"([0-9]+(?:\.[0-9]+)?)",
                log.read_text(errors="replace"),
                re.I,
            )

            if match:
                fps = float(match.group(1))
                source = "Analysis log.txt"

        if fps is None:
            times = pd.to_numeric(
                pd.Series(time_index),
                errors="coerce",
            ).to_numpy(dtype=float)

            if (
                len(times) < 2
                or not np.isfinite(times).all()
            ):
                raise ValueError(
                    f"Cannot infer fps from time/ID for {folder}; "
                    "supply explicit fps or a usable Analysis log.txt"
                )

            differences = np.diff(times)

            if not np.all(differences > 0):
                raise ValueError(
                    f"Invalid time/ID for {folder}: "
                    "duplicate or non-increasing timestamps"
                )

            if np.median(differences) >= 1:
                raise ValueError(
                    f"Ambiguous time/ID units for {folder}; "
                    "supply explicit fps or a usable Analysis log.txt"
                )

            if len(times) < 7:
                raise ValueError(
                    "Ambiguous short rounded time/ID index "
                    f"({len(times)} timestamps) for {folder}; "
                    "supply explicit fps or a usable Analysis log.txt"
                )

            frame_cap = 3000
            count = len(times)
            first_stamp = float(times[0])
            matching = []

            for candidate in range(2, 101):
                for first_frame in range(1, frame_cap + 1):
                    frame_time = round(
                        first_frame / candidate, 2
                    )
                    start_t = round(
                        frame_time - first_stamp, 2
                    )

                    if (
                        start_t < 0
                        or start_t > frame_time
                    ):
                        continue

                    if first_frame > 1:
                        previous = round(
                            (first_frame - 1) / candidate, 2
                        )
                        if not previous < start_t:
                            continue

                    if abs(
                        round(frame_time - start_t, 2)
                        - first_stamp
                    ) >= 1e-8:
                        continue

                    matched = True

                    for i in range(count):
                        expected = round(
                            round(
                                (first_frame + i) / candidate,
                                2,
                            ) - start_t,
                            2,
                        )

                        if abs(
                            expected - float(times[i])
                        ) >= 1e-8:
                            matched = False
                            break

                    if matched:
                        matching.append(candidate)
                        break

            if len(matching) != 1:
                raise ValueError(
                    f"Ambiguous time/ID fps for {folder}: "
                    f"{len(matching)} matching integer frame rates; "
                    "supply explicit fps or a usable Analysis log.txt"
                )

            fps = float(matching[0])
            source = "time/ID (centisecond-rounded)"

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    return fps, source


# ============================================================
# — PER-ANIMAL BOUT MEASUREMENTS
# ============================================================

def _bouts(cells, selected):
    """Extract bouts; NA and excluded behaviors break sequences."""
    bouts = []
    current = None
    length = 0
    valid_frames = 0

    for cell in cells:
        state = _event(cell)

        if state in selected:
            valid_frames += 1

            if state == current:
                length += 1
            else:
                if current is not None:
                    bouts.append((current, length))

                current = state
                length = 1
        else:
            if current is not None:
                bouts.append((current, length))

            current = None
            length = 0

    if current is not None:
        bouts.append((current, length))

    return bouts, valid_frames


def build_per_animal_measurements(
    recordings,
    output_folder=None,
    collapse_repeats=True,
    include_self=False,
    fps_tolerance=0.01,
):
    """Return per-animal transitions, behavior summaries, and bouts.

    Mixed FPS is allowed for per-animal descriptive exports.
    fps_tolerance is retained for backwards compatibility.
    """
    recordings = list(recordings)

    groups = {
        str(recording["group"])
        for recording in recordings
    }

    if not recordings or len(groups) != 2:
        raise ValueError(
            "Exactly two groups are required"
        )

    if fps_tolerance < 0:
        raise ValueError(
            "fps_tolerance must be nonnegative"
        )

    expected = set(recordings[0]["behaviors"])
    excluded = set(recordings[0].get("excluded", ()))

    transitions = []
    behaviors = []
    bout_rows = []
    seen = set()

    for recording in recordings:
        folder = Path(recording["folder"]).resolve()

        identity_key = (
            str(recording["group"]),
            str(folder),
        )

        if identity_key in seen:
            raise ValueError(
                f"Duplicate recording: {folder}"
            )

        seen.add(identity_key)

        if set(recording["behaviors"]) != expected:
            raise ValueError(
                "Incompatible behavior sets"
            )

        if set(recording.get("excluded", ())) != excluded:
            raise ValueError(
                "Incompatible exclusions"
            )

        selected = [
            behavior
            for behavior in recording["behaviors"]
            if behavior not in excluded
        ]

        if not selected:
            raise ValueError(
                "No included behaviors"
            )

        recording_transitions = load_recording(
            folder,
            recording["group"],
            recording["behaviors"],
            excluded,
            collapse_repeats,
            include_self,
        )

        workbook_paths = _workbooks(folder)

        for path, species in workbook_paths:
            df = pd.read_excel(path)

            if "time/ID" not in df.columns:
                raise ValueError(
                    f"Missing time/ID in {path}"
                )

            fps, fps_source = _fps(
                folder,
                df["time/ID"],
                recording.get("fps"),
            )

            for col in df.columns:
                if col == "time/ID":
                    continue

                animal_id = str(col)

                identity = {
                    "group": str(recording["group"]),
                    "source_recording": str(folder),
                    "species": species,
                    "animal_id": animal_id,
                }

                bouts, valid_frames = _bouts(
                    df[col],
                    set(selected),
                )

                for number, (state, length) in enumerate(
                    bouts, start=1
                ):
                    bout_rows.append({
                        **identity,
                        "behavior": state,
                        "bout_number": number,
                        "bout_length_frames": length,
                        "bout_duration_seconds": length / fps,
                        "fps": fps,
                        "fps_source": fps_source,
                    })

                if species is None:
                    species_mask = recording_transitions["species"].isna()
                else:
                    species_mask = recording_transitions["species"].eq(species)

                subset = recording_transitions[
                    (recording_transitions["animal_id"] == animal_id)
                    & species_mask
                ].copy()

                total_transitions = int(
                    subset["transition_count"].sum()
                )

                rate = (
                    1000 * total_transitions / valid_frames
                    if valid_frames
                    else np.nan
                )

                subset["valid_included_frames"] = valid_frames
                subset["animal_transition_count"] = total_transitions
                subset["switching_rate_per_1000_frames"] = rate
                subset["fps"] = fps
                subset["fps_source"] = fps_source

                transitions.append(subset)

                for behavior in selected:
                    lengths = [
                        length
                        for state, length in bouts
                        if state == behavior
                    ]

                    durations = (
                        np.asarray(lengths, dtype=float) / fps
                    )

                    behaviors.append({
                        **identity,
                        "behavior": behavior,
                        "valid_included_frames": valid_frames,
                        "animal_transition_count": total_transitions,
                        "switching_rate_per_1000_frames": rate,
                        "bout_count": len(lengths),
                        "total_bout_frames": sum(lengths),
                        "mean_bout_frames": (
                            float(np.mean(lengths))
                            if lengths else np.nan
                        ),
                        "median_bout_frames": (
                            float(np.median(lengths))
                            if lengths else np.nan
                        ),
                        "mean_bout_duration_seconds": (
                            float(np.mean(durations))
                            if lengths else np.nan
                        ),
                        "median_bout_duration_seconds": (
                            float(np.median(durations))
                            if lengths else np.nan
                        ),
                        "fps": fps,
                        "fps_source": fps_source,
                    })

    transition_df = pd.concat(
        transitions, ignore_index=True
    )

    behavior_df = pd.DataFrame(behaviors)

    bout_df = pd.DataFrame(
        bout_rows,
        columns=IDENTITY + [
            "behavior",
            "bout_number",
            "bout_length_frames",
            "bout_duration_seconds",
            "fps",
            "fps_source",
        ],
    )

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True, exist_ok=True
        )

        transition_df.to_csv(
            output_folder / "per_animal_transitions.csv",
            index=False,
        )
        behavior_df.to_csv(
            output_folder / "per_animal_behaviors.csv",
            index=False,
        )
        bout_df.to_csv(
            output_folder / "per_animal_bouts.csv",
            index=False,
        )

    return transition_df, behavior_df, bout_df


# ============================================================
# — VALIDATION AND SHARED ALPHABET
# ============================================================

def _validate_group_comparison(transition_df, behavior_df, group_order=None):
    """Validate per-animal tables and resolve a shared behavior alphabet."""
    if transition_df.empty or behavior_df.empty:
        raise ValueError("Per-animal input tables must not be empty")

    required_t = set(IDENTITY) | {
        "source_behavior",
        "target_behavior",
        "transition_count",
        "outgoing_count",
        "transition_probability",
        "probability_defined",
        "switching_rate_per_1000_frames",
    }
    required_b = set(IDENTITY) | {
        "behavior",
        "bout_count",
        "total_bout_frames",
        "mean_bout_frames",
        "median_bout_frames",
        "mean_bout_duration_seconds",
        "median_bout_duration_seconds",
        "switching_rate_per_1000_frames",
        "valid_included_frames",
        "animal_transition_count",
    }

    for label, frame, required in (
        ("transition", transition_df, required_t),
        ("behavior", behavior_df, required_b),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Missing {label} columns: {sorted(missing)}"
            )

    groups = list(pd.unique(transition_df["group"]))

    if len(groups) != 2:
        raise ValueError(
            "Comparative analysis requires exactly two groups"
        )

    if group_order is not None:
        requested = list(group_order)
        if (
            len(requested) != 2
            or set(requested) != set(groups)
        ):
            raise ValueError(
                "group_order must contain exactly the two groups"
            )
        groups = requested

    if set(behavior_df["group"]) != set(groups):
        raise ValueError("Group mismatch between per-animal tables")

    alphabet = list(pd.unique(behavior_df["behavior"]))

    if not alphabet:
        raise ValueError("Empty behavior alphabet")

    expected = set(alphabet)

    for group in groups:
        b = behavior_df[behavior_df["group"] == group]
        t = transition_df[transition_df["group"] == group]

        if set(b["behavior"]) != expected:
            raise ValueError(
                f"Incompatible behavior alphabet in {group}"
            )

        if (
            set(t["source_behavior"]) != expected
            or set(t["target_behavior"]) != expected
        ):
            # Self edges may be intentionally omitted, but every
            # behavior must still occur as a source and target.
            if len(expected) > 1:
                raise ValueError(
                    f"Incompatible transition alphabet in {group}"
                )

    # Identity includes recording and species, not just local animal ID.
    for frame, extra in (
        (
            transition_df,
            ["source_behavior", "target_behavior"],
        ),
        (behavior_df, ["behavior"]),
    ):
        keys = IDENTITY + extra

        if frame.duplicated(keys).any():
            raise ValueError(
                f"Duplicate per-animal observations: {keys}"
            )

    # Require the same set of animals in both per-animal exports.
    def identity_set(frame):
        return set(
            tuple(
                None if pd.isna(v) else v
                for v in row
            )
            for row in frame[IDENTITY].itertuples(
                index=False, name=None
            )
        )

    if identity_set(transition_df) != identity_set(behavior_df):
        raise ValueError(
            "Per-animal identities differ between per-animal tables"
        )

    return alphabet, groups


def _align_transition_edges(transition_df, alphabet, groups):
    """Construct the shared directed edge alphabet."""
    observed = set(
        zip(
            transition_df["source_behavior"],
            transition_df["target_behavior"],
        )
    )

    allowed = [
        (a, b)
        for a in alphabet
        for b in alphabet
        if (a, b) in observed
    ]

    # Each group must contain the same structural edges.
    for group in groups:
        frame = transition_df[
            transition_df["group"] == group
        ]
        group_edges = set(
            zip(
                frame["source_behavior"],
                frame["target_behavior"],
            )
        )

        if group_edges != observed:
            raise ValueError(
                "Incompatible transition edge sets between groups"
            )

    rows = [
        {
            "group": group,
            "source_behavior": a,
            "target_behavior": b,
        }
        for group in groups
        for a, b in allowed
    ]

    return pd.DataFrame(rows), allowed


def _summarize_animal_metrics(frame, group_columns, metrics):
    """Summarize animal-level metrics using equal animal weights.

    NaN values do not contribute to a mean. Observed zeros do.
    """
    records = []

    for keys, subset in frame.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys))

        row["n_animals"] = len(subset)

        for metric in metrics:
            values = pd.to_numeric(
                subset[metric],
                errors="coerce",
            )
            defined = values.notna()

            row[f"n_contributing_{metric}"] = int(
                defined.sum()
            )

            row[f"mean_{metric}"] = (
                float(values[defined].mean())
                if defined.any()
                else np.nan
            )

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# — GROUP SUMMARIES
# ============================================================

def top_group_transitions(transition_summary, groups, top_n=5):
    """Rank directed transitions separately by each group's mean probability.

    Undefined means are not ranked; defined zero-probability edges remain eligible. Equal means
    are resolved by source then target name for reproducible exports.
    """
    columns = ["group", "rank", "source_behavior", "target_behavior",
               "mean_transition_probability", "n_animals",
               "n_contributing_transition_probability"]
    result = []
    for group in groups:
        frame = transition_summary.loc[
            transition_summary["group"].eq(group)
        ].copy()
        frame["mean_transition_probability"] = pd.to_numeric(
            frame["mean_transition_probability"], errors="coerce")
        frame = frame.loc[
            np.isfinite(frame["mean_transition_probability"])
        ].sort_values(
            ["mean_transition_probability", "source_behavior", "target_behavior"],
            ascending=[False, True, True], kind="stable"
        ).head(top_n).copy()
        frame["rank"] = np.arange(1, len(frame) + 1)
        if "n_contributing_transition_probability" not in frame:
            frame["n_contributing_transition_probability"] = np.nan
        result.append(frame[columns])
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame(columns=columns)


def build_group_summaries(
    transition_df,
    behavior_df,
    group_order=None,
    behavior_order=None,
    output_folder=None,
    difference_direction="A-B",
):
    """Build descriptive group summaries without pooling animals.

    Transition probabilities:
        Mean of defined per-animal probabilities.

    Undefined probabilities:
        Excluded from the mean, retained as NaN.

    Signed difference:
        First group minus second group (default).

    Every summary retains n_animals and contributor counts.
    """
    if difference_direction not in ("A-B", "B-A"):
        raise ValueError("difference_direction must be A-B or B-A")
    alphabet, groups = _validate_group_comparison(
        transition_df,
        behavior_df,
        group_order,
    )

    if behavior_order is not None:
        order = list(behavior_order)

        if (
            len(order) != len(alphabet)
            or set(order) != set(alphabet)
        ):
            raise ValueError(
                "behavior_order must be a permutation "
                "of the shared alphabet"
            )

        alphabet = order

    edges_grid, edges = _align_transition_edges(
        transition_df,
        alphabet,
        groups,
    )

    edge_metrics = [
        "transition_probability",
        "transition_count",
        "outgoing_count",
        "switching_rate_per_1000_frames",
    ]

    edge_summary = _summarize_animal_metrics(
        transition_df,
        [
            "group",
            "source_behavior",
            "target_behavior",
        ],
        edge_metrics,
    )

    edge_summary = edges_grid.merge(
        edge_summary,
        on=[
            "group",
            "source_behavior",
            "target_behavior",
        ],
        how="left",
        validate="one_to_one",
    )

    if edge_summary["n_animals"].isna().any():
        raise ValueError(
            "Missing transition edges in one or more groups"
        )

    behavior_metrics = [
        "bout_count",
        "total_bout_frames",
        "mean_bout_frames",
        "median_bout_frames",
        "mean_bout_duration_seconds",
        "median_bout_duration_seconds",
        "switching_rate_per_1000_frames",
        "valid_included_frames",
        "animal_transition_count",
    ]

    behavior_summary = _summarize_animal_metrics(
        behavior_df,
        ["group", "behavior"],
        behavior_metrics,
    )

    behavior_grid = pd.MultiIndex.from_product(
        [groups, alphabet],
        names=["group", "behavior"],
    ).to_frame(index=False)

    behavior_summary = behavior_grid.merge(
        behavior_summary,
        on=["group", "behavior"],
        how="left",
        validate="one_to_one",
    )

    if behavior_summary["n_animals"].isna().any():
        raise ValueError(
            "Missing behavior rows in a group"
        )

    # Align group summaries before calculating differences.
    left = edge_summary[
        edge_summary["group"] == groups[0]
    ].drop(columns="group")

    right = edge_summary[
        edge_summary["group"] == groups[1]
    ].drop(columns="group")

    difference = left.merge(
        right,
        on=["source_behavior", "target_behavior"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )

    difference.insert(
        0, "first_group", groups[0]
    )
    difference.insert(
        1, "second_group", groups[1]
    )

    direction_label = (f"{groups[0]} - {groups[1]}" if difference_direction == "A-B"
                       else f"{groups[1]} - {groups[0]}")
    difference["difference_direction"] = direction_label
    for metric in edge_metrics:
        col = f"mean_{metric}"
        first = difference[f"{col}_first"]
        second = difference[f"{col}_second"]
        difference[f"difference_{metric}"] = (first - second if difference_direction == "A-B" else second - first)

    matrices = {}

    for group in groups:
        subset = edge_summary[
            edge_summary["group"] == group
        ]

        matrices[group] = subset.pivot(
            index="source_behavior",
            columns="target_behavior",
            values="mean_transition_probability",
        ).reindex(
            index=alphabet,
            columns=alphabet,
        )

    difference_matrix = difference.pivot(
        index="source_behavior",
        columns="target_behavior",
        values="difference_transition_probability",
    ).reindex(
        index=alphabet,
        columns=alphabet,
    )

    top_transitions = top_group_transitions(edge_summary, groups)

    result = {
        "top_transitions": top_transitions,
        "top_transitions_by_group": {group: top_transitions.loc[top_transitions["group"].eq(group)].copy() for group in groups},
        "groups": groups,
        "difference_direction": difference_direction,
        "difference_direction_label": direction_label,
        "per_animal_transitions": transition_df.copy(),
        "per_animal_behaviors": behavior_df.copy(),
        "behaviors": alphabet,
        "transition_summary": edge_summary,
        "behavior_summary": behavior_summary,
        "signed_differences": difference,
        "probability_matrices": matrices,
        "difference_matrix": difference_matrix,
    }

    if output_folder is not None:
        destination = Path(output_folder)
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        top_transitions.to_csv(destination / "group_top_5_transitions.csv", index=False)
        for i, group in enumerate(groups, 1):
            top_transitions.loc[top_transitions["group"].eq(group)].to_csv(
                destination / f"group_{i}_top_5_transitions.csv", index=False)

        edge_summary.to_csv(
            destination / "group_transition_summary.csv",
            index=False,
        )

        behavior_summary.to_csv(
            destination / "group_behavior_summary.csv",
            index=False,
        )

        difference.to_csv(
            destination / "group_signed_differences.csv",
            index=False,
        )

        for i, group in enumerate(groups, 1):
            matrices[group].to_csv(
                destination
                / f"group_{i}_probability_matrix.csv"
            )

        difference_matrix.rename_axis(index=f"source_behavior ({direction_label})").to_csv(
            destination / "signed_difference_matrix.csv"
        )

    return result


def set_difference_direction(comparison, direction="A-B"):
    """Return a new comparison with reversed display sign, preserving group summaries."""
    if direction not in ("A-B", "B-A"):
        raise ValueError("direction must be A-B or B-A")
    result = comparison.copy()
    old = comparison.get("difference_direction", "A-B")
    groups = comparison["groups"]
    label = f"{groups[0]} - {groups[1]}" if direction == "A-B" else f"{groups[1]} - {groups[0]}"
    result["difference_direction"] = direction
    result["difference_direction_label"] = label
    result["signed_differences"] = comparison["signed_differences"].copy()
    result["signed_differences"]["difference_direction"] = label
    result["difference_matrix"] = comparison["difference_matrix"].copy()
    if old != direction:
        columns = [c for c in result["signed_differences"] if c.startswith("difference_") and c != "difference_direction"]
        result["signed_differences"][columns] *= -1
        result["difference_matrix"] *= -1
    return result

def transition_confirmation_list(comparison, selected_pairs):
    """Ordered, user-readable confirmation rows for selected matrix cells."""
    allowed = set(zip(comparison["signed_differences"]["source_behavior"],
                      comparison["signed_differences"]["target_behavior"]))
    pairs = list(dict.fromkeys(tuple(pair) for pair in selected_pairs))
    if not set(pairs).issubset(allowed):
        raise ValueError("Unknown or disabled transitions")
    return pd.DataFrame([{"source_behavior": a, "target_behavior": b,
                          "transition": f"{a} → {b}"} for a, b in pairs],
                        columns=["source_behavior", "target_behavior", "transition"])

def selected_per_animal_values(comparison, selected_pairs):
    """Trace each selected measurement to its original animal and recording."""
    confirmation = transition_confirmation_list(comparison, selected_pairs)
    source = comparison.get("per_animal_transitions")
    if source is None:
        raise ValueError("Missing per-animal transitions")
    keys = set(zip(confirmation["source_behavior"], confirmation["target_behavior"]))
    mask = [(a,b) in keys for a,b in zip(source["source_behavior"], source["target_behavior"])]
    return source.loc[mask].copy().reset_index(drop=True)

def export_signed_differences(comparison, output_folder):
    """Export direction-labeled signed differences and matrix."""
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    label = comparison["difference_direction_label"]
    comparison["signed_differences"].to_csv(folder / "group_signed_differences.csv", index=False)
    comparison["difference_matrix"].rename_axis(index=f"source_behavior ({label})").to_csv(
        folder / "signed_difference_matrix.csv")


# ============================================================
# — TRANSITION SELECTION
# ============================================================

def select_transitions(comparison, selected_pairs, include_animals=False):
    """Return group summaries for selected directed transitions."""
    diff = comparison["signed_differences"]
    summary = comparison["transition_summary"]

    allowed = set(
        zip(
            diff["source_behavior"],
            diff["target_behavior"],
        )
    )

    selected = set(map(tuple, selected_pairs))

    if not selected.issubset(allowed):
        raise ValueError(
            f"Unknown or disabled transitions: "
            f"{selected - allowed}"
        )

    mask_d = [
        (a, b) in selected
        for a, b in zip(
            diff["source_behavior"],
            diff["target_behavior"],
        )
    ]

    mask_s = [
        (a, b) in selected
        for a, b in zip(
            summary["source_behavior"],
            summary["target_behavior"],
        )
    ]

    selected_summary = summary.loc[mask_s].copy()
    selected_diff = diff.loc[mask_d].copy()
    if include_animals:
        return selected_summary, selected_diff, selected_per_animal_values(comparison, selected_pairs)
    return selected_summary, selected_diff


# ============================================================
# — ALIGNED MATRIX VISUALIZATION
# ============================================================

def plot_comparison_matrices(
    comparison,
    output_path=None,
    selected_pairs=None,
):
    """Plot two group probability matrices and signed differences."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    groups = comparison["groups"]
    alphabet = comparison["behaviors"]

    matrices = [
        comparison["probability_matrices"][group]
        for group in groups
    ]

    matrices.append(
        comparison["difference_matrix"]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            max(12, len(alphabet) * 2.4),
            max(4, len(alphabet) * 0.6 + 2),
        ),
        constrained_layout=True,
    )

    delta_values = matrices[2].to_numpy(dtype=float)

    max_delta = (
        np.nanmax(np.abs(delta_values))
        if np.isfinite(delta_values).any()
        else 1.0
    )
    max_delta = max(max_delta, 1e-12)

    titles = [
        str(groups[0]),
        str(groups[1]),
        comparison.get("difference_direction_label", f"{groups[0]} - {groups[1]}"),
    ]

    selected = (
        None
        if selected_pairs is None
        else set(map(tuple, selected_pairs))
    )

    for i, (ax, matrix, title) in enumerate(
        zip(axes, matrices, titles)
    ):
        data = np.ma.masked_invalid(
            matrix.to_numpy(dtype=float)
        )

        cmap = plt.get_cmap(
            "RdBu_r" if i == 2 else "viridis"
        ).copy()
        cmap.set_bad(color="#dddddd")

        image = ax.imshow(
            data,
            cmap=cmap,
            vmin=-max_delta if i == 2 else 0,
            vmax=max_delta if i == 2 else 1,
        )

        ax.set_title(title)

        ax.set_xticks(
            range(len(alphabet)),
            alphabet,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            range(len(alphabet)),
            alphabet,
        )

        ax.set_xlabel("Target behavior")
        ax.set_ylabel("Source behavior")

        if selected is not None:
            for r, source in enumerate(alphabet):
                for c, target in enumerate(alphabet):
                    if (source, target) in selected:
                        ax.add_patch(
                            Rectangle(
                                (c - 0.5, r - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                            )
                        )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.7,
        )
        if i == 2:
            colorbar.set_label(titles[2])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=180,
        )

    return fig


# ============================================================
# — INTERACTIVE TRANSITION MATRIX
# ============================================================

def launch_transition_selector(comparison):
    """Open an interactive matrix for selecting directed transitions.

    Click a cell to select or deselect its directed edge.
    Blue cells indicate selected transitions.

    Returns:
        figure, selected_pairs

    Requires an interactive Matplotlib backend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from matplotlib.patches import Rectangle

    alphabet = comparison["behaviors"]

    diff = comparison["signed_differences"]

    available = set(
        zip(
            diff["source_behavior"],
            diff["target_behavior"],
        )
    )

    selected = set()

    fig, ax = plt.subplots(
        figsize=(
            max(6, len(alphabet) * 0.75 + 2),
            max(5, len(alphabet) * 0.75 + 2),
        )
    )

    fig.subplots_adjust(bottom=0.29)

    ax.set_xlim(
        -0.5,
        len(alphabet) - 0.5,
    )

    ax.set_ylim(
        len(alphabet) - 0.5,
        -0.5,
    )

    ax.set_xticks(
        range(len(alphabet)),
        alphabet,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(alphabet)),
        alphabet,
    )

    ax.set_xlabel(
        "Target behavior (click a cell)"
    )
    ax.set_ylabel("Source behavior")
    ax.set_title("Select directed transitions")

    patches = {}

    for r, source in enumerate(alphabet):
        for c, target in enumerate(alphabet):
            pair = (source, target)

            color = (
                "white"
                if pair in available
                else "#dddddd"
            )

            patch = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="#999999",
            )

            ax.add_patch(patch)
            patches[pair] = patch

    confirmation = fig.text(0.04, 0.13, "Selected transitions: none", fontsize=9, va="top")

    def click(event):
        if (
            event.inaxes is not ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        c = int(round(event.xdata))
        r = int(round(event.ydata))

        if not (
            0 <= r < len(alphabet)
            and 0 <= c < len(alphabet)
        ):
            return

        pair = (
            alphabet[r],
            alphabet[c],
        )

        if pair not in available:
            return

        if pair in selected:
            selected.remove(pair)
        else:
            selected.add(pair)

        patches[pair].set_facecolor(
            "#4da6ff"
            if pair in selected
            else "white"
        )

        names = transition_confirmation_list(comparison, sorted(selected))["transition"].tolist()
        confirmation.set_text("Selected transitions: " + (", ".join(names) if names else "none"))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "button_press_event",
        click,
    )

    done_ax = fig.add_axes([
        0.78, 0.03, 0.16, 0.07
    ])

    done_button = Button(
        done_ax,
        "Done",
    )

    def done(_event):
        plt.close(fig)

    done_button.on_clicked(done)

    # Keep a reference so the widget is not garbage-collected.
    fig._transition_selector_button = done_button

    plt.show()

    return fig, selected


# ============================================================
#
# ============================================================

def build_comparative_analysis(
    recordings,
    output_folder=None,
    group_order=None,
    behavior_order=None,
    collapse_repeats=True,
    include_self=False,
    difference_direction="A-B",
):
    """Run the comparative transition analysis pipeline.

    Parameters
    ----------
    recordings : list of dict
        Each dictionary describes one analyzed video:
        folder, group, behaviors, excluded, optional fps.

    output_folder : str or Path, optional
        Folder for per-animal and group-level CSV exports.

    group_order : tuple, optional
        Defines the subtraction order:
        first group minus second group by default.

    behavior_order : list, optional
        Common ordering for all group matrices.

    Returns
    -------
    dict
        Per-animal tables, group summaries, aligned probability
        matrices, and signed differences.
    """
    per_animal_folder = (
        None
        if output_folder is None
        else Path(output_folder)
    )

    transitions, behaviors, bouts = (
        build_per_animal_measurements(
            recordings,
            output_folder=per_animal_folder,
            collapse_repeats=collapse_repeats,
            include_self=include_self,
        )
    )

    comparison = build_group_summaries(
        transitions,
        behaviors,
        group_order=group_order,
        behavior_order=behavior_order,
        output_folder=output_folder,
        difference_direction=difference_direction,
    )

    comparison["per_animal_transitions"] = transitions
    comparison["per_animal_behaviors"] = behaviors
    comparison["per_animal_bouts"] = bouts

    return comparison


# ============================================================
# — CONFIRMED TWO-GROUP PERMUTATION STATISTICS
# ============================================================
# This path is independent of Mine Results statistics.
# Dependencies: scipy, statsmodels (direct project dependencies).

import math
from scipy import stats as _scipy_stats
from statsmodels.stats.multitest import multipletests as _multipletests

_TRANSITION_TEST_METRICS = {
    "transition_count", "transition_probability", "switching_rate_per_1000_frames",
}
_BEHAVIOR_TEST_METRICS = {
    "mean_bout_duration_seconds", "median_bout_duration_seconds",
    "switching_rate_per_1000_frames",
}
_ALLOWED_TEST_METRICS = _TRANSITION_TEST_METRICS | _BEHAVIOR_TEST_METRICS
_ANIMAL_IDENTITY_COLUMNS = IDENTITY


def _normalize_confirmed_selection(selection):
    """Normalize one user-confirmed measurement specification.

    Accepted forms:
      {"kind": "transition", "source_behavior": "A", "target_behavior": "B",
       "metric": "transition_probability"}
      {"kind": "behavior", "behavior": "A", "metric": "mean_bout_duration_seconds"}
      ("A", "B")  # shorthand for a transition probability

    A selection is never inferred from all available rows.
    """
    if isinstance(selection, (tuple, list)) and len(selection) == 2:
        selection = {"kind": "transition", "source_behavior": selection[0],
                     "target_behavior": selection[1], "metric": "transition_probability"}
    if not isinstance(selection, dict):
        raise ValueError("Each confirmed selection must be a dict or a directed pair")
    item = dict(selection)
    kind = item.get("kind", "transition")
    metric = item.get("metric", "transition_probability" if kind == "transition" else None)
    if kind not in ("transition", "behavior"):
        raise ValueError("Selection kind must be 'transition' or 'behavior'")
    allowed = _TRANSITION_TEST_METRICS if kind == "transition" else _BEHAVIOR_TEST_METRICS
    if metric not in allowed:
        raise ValueError(f"Unsupported {kind} metric: {metric!r}")
    if kind == "transition":
        if not item.get("source_behavior") or not item.get("target_behavior"):
            raise ValueError("Transition selection needs source_behavior and target_behavior")
        label = f"{item['source_behavior']} → {item['target_behavior']} | {metric}"
    else:
        if not item.get("behavior"):
            raise ValueError("Behavior selection needs behavior")
        label = f"{item['behavior']} | {metric}"
    return {"kind": kind, "metric": metric,
            "source_behavior": item.get("source_behavior") if kind == "transition" else None,
            "target_behavior": item.get("target_behavior") if kind == "transition" else None,
            "behavior": item.get("behavior") if kind == "behavior" else None,
            "selection_label": label}


def _cliffs_delta(a, b):
    """Cliff's delta: positive means Group A tends to exceed Group B."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        return np.nan
    return float((np.count_nonzero(a[:, None] > b[None, :]) -
                  np.count_nonzero(a[:, None] < b[None, :])) / (len(a) * len(b)))


def _permutation_test(a, b, seed=0, exact_limit=100000, random_assignments=99999):
    """SciPy two-sided independent permutation test of mean(A) - mean(B).

    Exact if there are at most 100,000 distinct assignments; otherwise
    99,999 random resamples plus the observed arrangement (100,000 total).
    SciPy's two-sided p-value is twice the smaller one-sided tail,
    capped at one, rather than the absolute-statistic tail probability.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not len(a) or not len(b) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Permutation samples must be nonempty and finite")
    assignments = math.comb(len(a) + len(b), len(a))
    exact = assignments <= exact_limit

    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    test = _scipy_stats.permutation_test(
        (a, b), statistic,
        permutation_type="independent",
        alternative="two-sided",
        n_resamples=np.inf if exact else random_assignments,
        random_state=None if exact else int(seed),
        vectorized=False,
    )
    return (float(test.pvalue), "exact" if exact else "randomized",
            int(assignments if exact else random_assignments + 1),
            None if exact else int(seed))


def _values_for_confirmed_selection(comparison, spec):
    """Return per-animal source rows, preserving recording identity and FPS."""
    if spec["kind"] == "transition":
        frame = comparison["per_animal_transitions"]
        subset = frame.loc[
            frame["source_behavior"].eq(spec["source_behavior"]) &
            frame["target_behavior"].eq(spec["target_behavior"])
        ].copy()
    else:
        frame = comparison["per_animal_behaviors"]
        subset = frame.loc[frame["behavior"].eq(spec["behavior"])].copy()
    if subset.empty:
        raise ValueError(f"Confirmed measurement does not exist: {spec['selection_label']}")
    if spec["metric"] not in subset.columns:
        raise ValueError(f"Missing measurement column: {spec['metric']}")
    if subset.duplicated(_ANIMAL_IDENTITY_COLUMNS).any():
        raise ValueError(f"Duplicate animal-level observations: {spec['selection_label']}")
    subset["measurement_value"] = pd.to_numeric(subset[spec["metric"]], errors="coerce")
    subset["excluded_from_test"] = ~np.isfinite(subset["measurement_value"].to_numpy(dtype=float))
    subset["exclusion_reason"] = np.where(subset["excluded_from_test"], "undefined_or_nonfinite", "")
    subset["selection_label"] = spec["selection_label"]
    subset["measurement_metric"] = spec["metric"]
    subset["measurement_kind"] = spec["kind"]
    return subset


def build_two_group_statistics(
    comparison, confirmed_selections, output_folder=None, seed=0,
    alpha=0.05, min_animals=5, fps_tolerance=0.01,
):
    """Test only confirmed measurements using independent animals.

    Group A/B follow comparison['groups'], independent of display sign.
    Undefined probabilities are excluded; observed zeroes remain valid.
    Switching rates are blocked only if contributing animals have
    incompatible FPS. Holm is applied to all testable selections in this run.
    All selections (including non-testable) are returned and exported.

    Returns dict with 'results' and 'per_animal_values' DataFrames.
    """
    if not 0 < alpha < 1 or min_animals < 1 or fps_tolerance < 0:
        raise ValueError("Invalid alpha, min_animals, or fps_tolerance")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    groups = list(comparison["groups"])
    if len(groups) != 2 or groups[0] == groups[1]:
        raise ValueError("Exactly two distinct groups are required")
    selections = [_normalize_confirmed_selection(x) for x in confirmed_selections]
    labels = [x["selection_label"] for x in selections]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate confirmed selections")
    rows, animal_rows = [], []
    for spec in selections:
        subset = _values_for_confirmed_selection(comparison, spec)
        if not set(subset["group"]).issubset(set(groups)):
            raise ValueError("Unexpected group in per-animal observations")
        subset["group_role"] = subset["group"].map({groups[0]: "A", groups[1]: "B"})
        animal_rows.append(subset)
        values = []
        n_total, n_excluded, n_valid = [], [], []
        for group in groups:
            group_frame = subset.loc[subset["group"].eq(group)]
            valid = group_frame.loc[~group_frame["excluded_from_test"]]
            values.append(valid["measurement_value"].to_numpy(dtype=float))
            n_total.append(len(group_frame))
            n_valid.append(len(valid))
            n_excluded.append(len(group_frame) - len(valid))
        a, b = values
        warnings = []
        reason = ""
        if min(n_valid) < min_animals:
            reason = f"insufficient_animals: at least {min_animals} contributing animals per group required"
        if spec["metric"] == "switching_rate_per_1000_frames":
            valid_fps = subset.loc[~subset["excluded_from_test"], "fps"] if "fps" in subset else pd.Series(dtype=float)
            fps = pd.to_numeric(valid_fps, errors="coerce").to_numpy(dtype=float)
            if len(fps) != sum(n_valid) or not np.isfinite(fps).all() or (len(fps) and np.ptp(fps) > fps_tolerance):
                reason = (reason + "; " if reason else "") + "incompatible_fps_for_switching_rate"
        if spec["metric"] == "transition_count":
            warnings.append("raw_counts_not_exposure_normalized")
        if sum(n_excluded):
            warnings.append("undefined_or_nonfinite_values_excluded")
        row = {**spec, "group_a": groups[0], "group_b": groups[1],
               "difference_direction": f"{groups[0]} - {groups[1]}",
               "n_total_a": n_total[0], "n_total_b": n_total[1],
               "n_contributing_a": n_valid[0], "n_contributing_b": n_valid[1],
               "n_excluded_a": n_excluded[0], "n_excluded_b": n_excluded[1],
               "mean_a": float(a.mean()) if len(a) else np.nan,
               "mean_b": float(b.mean()) if len(b) else np.nan,
               "mean_difference_a_minus_b": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
               "cliffs_delta_a_vs_b": _cliffs_delta(a, b),
               "testable": not bool(reason), "non_testable_reason": reason,
               "p_raw": np.nan, "p_holm": np.nan, "significant_holm": None,
               "alpha": alpha, "test_type": "not_testable", "permutation_count": 0,
               "seed": np.nan, "warnings": "; ".join(warnings)}
        if not reason:
            p, method, count, used_seed = _permutation_test(a, b, seed=int(seed))
            row.update(p_raw=p, test_type=method, permutation_count=count,
                       seed=used_seed if used_seed is not None else np.nan)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        eligible = result["testable"].astype(bool)
        if eligible.any():
            rejected, corrected, _, _ = _multipletests(
                result.loc[eligible, "p_raw"].to_numpy(dtype=float),
                alpha=alpha, method="holm")
            result.loc[eligible, "p_holm"] = corrected
            result.loc[eligible, "significant_holm"] = pd.Series(
                [bool(x) for x in rejected], index=result.index[eligible], dtype=object
            )
        # Keep non-testable significance genuinely missing, not False.
        result["significant_holm"] = result["significant_holm"].astype(object)
        result.loc[~eligible, "significant_holm"] = None
    animals = pd.concat(animal_rows, ignore_index=True) if animal_rows else pd.DataFrame()
    output = {"results": result, "per_animal_values": animals,
              "groups": groups, "alpha": alpha, "random_seed": int(seed)}
    if output_folder is not None:
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        result.to_csv(folder / "two_group_permutation_statistics.csv", index=False)
        animals.to_csv(folder / "two_group_selected_per_animal_values.csv", index=False)
    return output


def build_comparative_analysis_with_statistics(
    recordings, confirmed_selections, output_folder=None, seed=0,
    group_order=None, behavior_order=None, collapse_repeats=True,
    include_self=False, difference_direction="A-B", alpha=0.05,
    min_animals=5, fps_tolerance=0.01,
):
    """Run comparative analysis with confirmed-selection permutation inference."""
    comparison = build_comparative_analysis(
        recordings, output_folder=output_folder, group_order=group_order,
        behavior_order=behavior_order, collapse_repeats=collapse_repeats,
        include_self=include_self, difference_direction=difference_direction)
    comparison["two_group_statistics"] = build_two_group_statistics(
        comparison, confirmed_selections, output_folder=output_folder,
        seed=seed, alpha=alpha, min_animals=min_animals,
        fps_tolerance=fps_tolerance)
    return comparison
