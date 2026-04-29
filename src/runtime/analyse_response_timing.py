from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_all_events(log_path: str) -> list[dict]:
    events = []
    decoder = json.JSONDecoder()

    with Path(log_path).open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            text = raw_line.strip()
            while text:
                try:
                    event, idx = decoder.raw_decode(text)
                    events.append(event)
                    text = text[idx:].strip()
                except json.JSONDecodeError:
                    break

    return events

def match_response_timings_to_tts(events: list[dict]) -> pd.DataFrame:
    granted_timings = []
    tts_events = []

    for event in events:
        event_type = event.get("event_type")
        timestamp_ns = event.get("timestamp_ns")
        data = event.get("data", {})

        if event_type == "response_timing" and data.get("permission_granted", False):
            granted_timings.append({
                **data,
                "event_timestamp_ns": timestamp_ns,
            })

        elif event_type == "tts_complete":
            tts_events.append({
                "tts_timestamp_ns": timestamp_ns,
                **data,
            })

    rows = []
    used_timing_ids = set()

    for tts in tts_events:
        candidates = [
            timing for timing in granted_timings
            if timing["event_timestamp_ns"] <= tts["tts_timestamp_ns"]
            and timing["event_timestamp_ns"] not in used_timing_ids
        ]

        if not candidates:
            continue

        best = max(candidates, key=lambda row: row["event_timestamp_ns"])
        used_timing_ids.add(best["event_timestamp_ns"])

        rows.append({
            **best,
            "tts_timestamp_ns": tts["tts_timestamp_ns"],
            "tts_synthesis_time_s": tts.get("synthesis_time_s"),
        })

    return pd.DataFrame(rows)

def inspect_log(log_path: str) -> None:
    decoder = json.JSONDecoder()
    total = 0
    timing = 0
    granted = 0
    modes = {}

    with Path(log_path).open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            text = raw_line.strip()
            while text:
                try:
                    event, idx = decoder.raw_decode(text)
                    text = text[idx:].strip()
                except json.JSONDecodeError:
                    break

                total += 1

                if event.get("event_type") == "response_timing":
                    timing += 1
                    data = event.get("data", {})
                    if data.get("permission_granted", False):
                        granted += 1
                    mode = data.get("mode", "MISSING")
                    modes[mode] = modes.get(mode, 0) + 1

    print(f"\nInspecting {log_path}")
    print(f"total events: {total}")
    print(f"response_timing events: {timing}")
    print(f"permission_granted response_timing events: {granted}")
    print(f"modes: {modes}")


def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    baseline_path = Path("logs/baseline_runtime_events.jsonl")
    gaze_path = Path("logs/gaze_runtime_events.jsonl")
    print(f"[PATH] baseline_path={baseline_path}")
    print(f"[PATH] baseline_exists={baseline_path.exists()}")

    print(f"[PATH] gaze_path={gaze_path}")
    print(f"[PATH] gaze_exists={gaze_path.exists()}")

    inspect_log(str(baseline_path))
    inspect_log(str(gaze_path))

    df_list = []

    if baseline_path.exists():
        events_baseline = load_all_events(str(baseline_path))
        df_baseline = match_response_timings_to_tts(events_baseline)
        df_baseline["mode"] = "baseline"
        df_list.append(df_baseline)

    if gaze_path.exists():
        events_gaze = load_all_events(str(gaze_path))
        df_gaze = match_response_timings_to_tts(events_gaze)
        df_gaze["mode"] = "gaze"
        df_list.append(df_gaze)

    if not df_list:
        raise RuntimeError(
            "No response timing logs found. Expected one or both of:\n"
            "logs/baseline_runtime_events.jsonl\n"
            "logs/gaze_runtime_events.jsonl"
        )

    df = pd.concat(df_list, ignore_index=True)

    required_columns = [
        "mode",
        "speech_end_time_ns",
        "response_start_time_ns",
        "response_latency_ms",
        "model_confidence",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in response_timing logs: {missing}")

    print("\nSample counts:")
    print(df["mode"].value_counts())

    summary = (
        df.groupby("mode")["response_latency_ms"]
        .agg(["mean", "count", "std", "min", "max"])
        .reset_index()
    )

    print("\nLatency summary:")
    print(summary)

    df.to_csv(output_dir / "response_timing_events.csv", index=False)
    summary.to_csv(output_dir / "response_latency_summary.csv", index=False)

    # Plot 1: mean latency comparison
    means = df.groupby("mode")["response_latency_ms"].mean()
    means.plot(kind="bar")
    plt.ylabel("Mean response latency (ms)")
    plt.xlabel("Mode")
    plt.title("Mean Response Latency: Baseline vs Gaze-Informed")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison_mean.png", dpi=300)
    plt.close()

    # Plot 2: box plot
    df.boxplot(column="response_latency_ms", by="mode")
    plt.suptitle("")
    plt.title("Response Latency Distribution by Mode")
    plt.ylabel("Response latency (ms)")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_boxplot.png", dpi=300)
    plt.close()

    # Plot 3: confidence vs latency, gaze only
    gaze_df = df[df["mode"] == "gaze"]

    if not gaze_df.empty:
        plt.scatter(gaze_df["model_confidence"], gaze_df["response_latency_ms"])
        plt.xlabel("Model confidence")
        plt.ylabel("Response latency (ms)")
        plt.title("Model Confidence vs Response Latency")
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_vs_latency.png", dpi=300)
        plt.close()

    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()