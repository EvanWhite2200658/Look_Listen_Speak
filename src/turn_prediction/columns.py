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

# current 24 feature set
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

# richer feature set candidate
RICH_TURN_PREDICTION_FEATURE_COLUMNS = [
    *CORE_GAZE_FEATURE_COLUMNS,
    *DELTA_GAZE_FEATURE_COLUMNS,
    "speaking",
    "target0.speaking",
    "turn_shift",
    "target0.x",
    "target0.y",
    "target0.z",
]

RUNTIME_COMPATIBLE_FEATURE_COLUMNS = [
    *CORE_GAZE_FEATURE_COLUMNS,
    *DELTA_GAZE_FEATURE_COLUMNS,
]

ALL_REQUIRED_PROCESSED_COLUMNS = [
    "id",
    "frameIndex",
    *RICH_TURN_PREDICTION_FEATURE_COLUMNS,
    "action",
    "vfoa_GT",
    "target0.name",
    MAIN_TARGET_COLUMN,
]