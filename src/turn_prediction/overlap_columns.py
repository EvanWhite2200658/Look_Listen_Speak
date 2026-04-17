# src/turn_prediction/overlap_columns.py
# explicit column contract for training on existing dataset

BASE_COLUMNS = [
    "eye.x", "eye.y", "eye.z",
    "gaze.x", "gaze.y", "gaze.z",
    "head.x", "head.y", "head.z",
    "headpose.roll", "headpose.pitch", "headpose.yaw",
]

DELTA_COLUMNS = [f"delta_{c}" for c in BASE_COLUMNS]

ALL_COLUMNS = BASE_COLUMNS + DELTA_COLUMNS

TARGET_COLUMN = "turn_shift_in_next_15_frames"