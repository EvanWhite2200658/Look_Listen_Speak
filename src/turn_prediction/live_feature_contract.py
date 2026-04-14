# src/turn_prediction/live_feature_contract.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.gaze.schemas import GazeSample
from columns import RICH_TURN_PREDICTION_FEATURE_COLUMNS


@dataclass(frozen=True)
class LiveFeatureContractReport:
    required_feature_names: list[str]
    available_live_fields: list[str]
    unsupported_feature_names: list[str]
    is_supported: bool


def get_required_rich_feature_names() -> list[str]:
    return list(RICH_TURN_PREDICTION_FEATURE_COLUMNS)


def get_available_live_fields() -> list[str]:
    """
    fields currently represented in project-local live gaze samples.
    these reflect src/gaze/schemas.py, not the dataset CSV schema.
    :return:
    """
    return [
        "timestamp_ns",
        "raw_xy",
        "calibrated_xy",
        "filtered_xy",
        "left_eye_openness",
        "right_eye_openness",
        "tracking_state",
        "status",
        "features",
    ]

def build_live_feature_contract_report() -> LiveFeatureContractReport:
    required = get_required_rich_feature_names()
    available = get_available_live_fields()

    unsupported = list(required)

    return LiveFeatureContractReport(
        required_feature_names=required,
        available_live_fields=available,
        unsupported_feature_names=unsupported,
        is_supported=(len(unsupported) == 0),
    )

def format_live_feature_contract_report(report: LiveFeatureContractReport) -> str:
    lines: list[str] = []
    lines.append("Live Feature Contract Report")
    lines.append(f"Supported: {report.is_supported}")
    lines.append("")
    lines.append("Required rich checkpoint features:")
    for name in report.required_feature_names:
        lines.append(f"- {name}")

    lines.append("")
    lines.append("Available live sample fields:")
    for name in report.available_live_fields:
        lines.append(f"- {name}")

    lines.append("")
    lines.append("Unsupported required features:")
    for name in report.unsupported_feature_names:
        lines.append(f"- {name}")

    return "\n".join(lines)