
"""LabGym comparative transition analysis: Checkpoints 1 and 2.

C1: Per-recording, per-animal bout-to-bout transitions.
C2: Per-animal switching rates, bout durations, and tidy exports.

No pooling across animals and no group-level statistics.
"""

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
# CHECKPOINT 1: INPUTS AND TRANSITION COUNTS
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
    """Discover only per-video probability workbooks."""
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
    """Load one analyzed video, preserving animal and species identity."""
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
                    "source_recording": str(
                        folder.resolve()
                    ),
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
    excluded = set(
        recordings[0].get("excluded", ())
    )

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
# CHECKPOINT 2: FPS INFERENCE
# ============================================================

def _fps(folder, time_index, explicit=None):
    """Resolve FPS from explicit input, analysis log, or frame clock.

    The time/ID fallback reproduces LabGym's writer: round each
    frame time to centiseconds, round the analysis start to
    centiseconds, subtract, and round that difference.

    Ambiguous candidates are rejected.
    """
    folder = Path(folder)

    # Priority 1: explicitly supplied FPS
    if explicit is not None:
        fps = float(explicit)
        source = "explicit"

    else:
        fps = None
        source = None

        # Priority 2: Analysis log.txt
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

        # Priority 3: LabGym time/ID frame clock
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

            # Conservative handling of short rounded indexes.
            if len(times) < 7:
                raise ValueError(
                    "Ambiguous short rounded time/ID index "
                    f"({len(times)} timestamps) for {folder}; "
                    "supply explicit fps or a usable Analysis log.txt"
                )

            # Writer frame numbers are 1-based. Past this cap,
            # a clock that would have matched fails closed.
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
                            (first_frame - 1) / candidate,
                            2,
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
                            )
                            - start_t,
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

            # Never guess between multiple possible FPS.
            if len(matching) != 1:
                raise ValueError(
                    f"Ambiguous time/ID fps for {folder}: "
                    f"{len(matching)} matching integer frame rates; "
                    "supply explicit fps or a usable Analysis log.txt"
                )

            fps = float(matching[0])
            source = "time/ID (centisecond-rounded)"

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            "fps must be finite and positive"
        )

    return fps, source


# ============================================================
# CHECKPOINT 2: BOUT MEASUREMENTS
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
                    bouts.append(
                        (current, length)
                    )

                current = state
                length = 1

        else:
            if current is not None:
                bouts.append(
                    (current, length)
                )

            current = None
            length = 0

    if current is not None:
        bouts.append(
            (current, length)
        )

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

    expected = set(
        recordings[0]["behaviors"]
    )

    excluded = set(
        recordings[0].get("excluded", ())
    )

    transitions = []
    behaviors = []
    bout_rows = []
    seen = set()

    for recording in recordings:
        folder = Path(
            recording["folder"]
        ).resolve()

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

        if set(
            recording.get("excluded", ())
        ) != excluded:
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

        # Reuse C1 counting logic.
        c1 = load_recording(
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

                # Individual bout export
                for number, (
                    state, length
                ) in enumerate(bouts, start=1):

                    bout_rows.append({
                        **identity,
                        "behavior": state,
                        "bout_number": number,
                        "bout_length_frames": length,
                        "bout_duration_seconds": (
                            length / fps
                        ),
                        "fps": fps,
                        "fps_source": fps_source,
                    })

                # Match C1 rows for this exact animal.
                if species is None:
                    species_mask = (
                        c1["species"].isna()
                    )
                else:
                    species_mask = (
                        c1["species"].eq(species)
                    )

                subset = c1[
                    (c1["animal_id"] == animal_id)
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

                subset["valid_included_frames"] = (
                    valid_frames
                )

                subset["animal_transition_count"] = (
                    total_transitions
                )

                subset[
                    "switching_rate_per_1000_frames"
                ] = rate

                subset["fps"] = fps
                subset["fps_source"] = fps_source

                transitions.append(subset)

                # Per-behavior summary for this animal
                for behavior in selected:
                    lengths = [
                        length
                        for state, length in bouts
                        if state == behavior
                    ]

                    durations = (
                        np.asarray(
                            lengths, dtype=float
                        ) / fps
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
        transitions,
        ignore_index=True,
    )

    behavior_df = pd.DataFrame(
        behaviors
    )

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
            parents=True,
            exist_ok=True,
        )

        transition_df.to_csv(
            output_folder
            / "per_animal_transitions.csv",
            index=False,
        )

        behavior_df.to_csv(
            output_folder
            / "per_animal_behaviors.csv",
            index=False,
        )

        bout_df.to_csv(
            output_folder
            / "per_animal_bouts.csv",
            index=False,
        )

    return (
        transition_df,
        behavior_df,
        bout_df,
    )
