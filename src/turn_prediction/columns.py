# src/turn_prediction/columns.py

from __future__ import annotations

CORE_GAZE_FEATURE_COLUMNS = [
    "eye.x",
    "eye.y",
    "eye.z",
    "gaze.x",
    "gaze.y",
    "gaze.z",
    "head.x",
    "head.y",
    "head.z",
    "headpose.roll",
    "headpose.pitch",
    "headpose.yaw",
]

DELTA_GAZE_FEATURE_COLUMNS = [
    "delta_eye.x",
    "delta_eye.y",
    "delta_eye.z",
    "delta_gaze.x",
    "delta_gaze.y",
    "delta_gaze.z",
    "delta_head.x",
    "delta_head.y",
    "delta_head.z",
    "delta_headpose.roll",
    "delta_headpose.pitch",
    "delta_headpose.yaw",
]

GAZE_TRANSFORMER_FEATURE_COLUMNS = (
    *CORE_GAZE_FEATURE_COLUMNS,
    *DELTA_GAZE_FEATURE_COLUMNS,
)

OPTIONAL_CONTEXT_COLUMNS = [
    "speaking",
    "target0.speaking",
    "speech_state",
]

REFERENCE_LABEL_COLUMNS = [
    "turn_shift",
]

MAIN_TARGET_COLUMN = "turn_shift_in_next_15_frames"

ALL_REQUIRED_PROCESSED_COLUMNS = [
    "id",
    "frameIndex",
    *GAZE_TRANSFORMER_FEATURE_COLUMNS,
    *OPTIONAL_CONTEXT_COLUMNS,
    "action",
    "vfoa_GT",
    "target0.name",
    "target0.x",
    "target0.y",
    "target0.z",
    *REFERENCE_LABEL_COLUMNS,
    MAIN_TARGET_COLUMN,
]