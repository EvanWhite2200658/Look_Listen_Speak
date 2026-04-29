from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_response_timing(log_path: str) -> pd.DataFrame:
    rows = []

    with Path(log_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue

            event = json.loads(line)
            if event.get("event_type") != "response_timing":
                continue

            data = event["data"]
            if not data.get("permission_granted", False):
                continue

            rows.append(data)

    return pd.DataFrame(rows)


def main() -> None:
    log_path = "logs/runtime_events.jsonl"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    df = load_response_timing(log_path)

    summary = (
        df.groupby("mode")["response_latency_ms"]
        .agg(["mean", "count", "std", "min", "max"])
        .reset_index()
    )

    print(summary)
    summary.to_csv(output_dir / "response_latency_summary.csv", index=False)
    df.to_csv(output_dir / "response_timing_events.csv", index=False)

    # Plot 1: latency comparison
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
    plt.scatter(gaze_df["model_confidence"], gaze_df["response_latency_ms"])
    plt.xlabel("Model confidence")
    plt.ylabel("Response latency (ms)")
    plt.title("Model Confidence vs Response Latency")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_vs_latency.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()