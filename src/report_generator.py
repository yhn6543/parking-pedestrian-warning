import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_KEYS = [
    "mode",
    "processed_frames",
    "raw_risk_frames",
    "stable_risk_frames",
    "logged_alert_count",
    "max_risk_person_count",
    "window_size",
    "min_risk_count",
    "cooldown_frames",
]


def load_json_file(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {json_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")

    return data


def load_csv_log(path: str):
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarize_alert_summary(summary: dict) -> dict:
    metrics = {}
    for key in SUMMARY_KEYS:
        default = "unknown" if key == "mode" else 0
        metrics[key] = summary.get(key, default)
    return metrics


def calculate_rates(summary_metrics: dict) -> dict:
    processed_frames = int(summary_metrics.get("processed_frames", 0) or 0)
    if processed_frames <= 0:
        return {
            "raw_risk_rate": 0.0,
            "stable_risk_rate": 0.0,
            "alert_log_rate": 0.0,
        }

    return {
        "raw_risk_rate": float(summary_metrics.get("raw_risk_frames", 0)) / processed_frames,
        "stable_risk_rate": float(summary_metrics.get("stable_risk_frames", 0)) / processed_frames,
        "alert_log_rate": float(summary_metrics.get("logged_alert_count", 0)) / processed_frames,
    }


def generate_markdown_report(metrics: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    generated_files = metrics.get("generated_files", [])
    generated_files_text = "\n".join(f"- `{item}`" for item in generated_files) or "- No output files listed."

    markdown = f"""# Parking Pedestrian Warning - Project Summary

## Pipeline Summary

This project detects `person` and `car` objects with YOLO, checks whether a pedestrian anchor point is inside a danger-zone ROI, stabilizes warnings across consecutive frames, and stores alert captures plus CSV logs.

## Completed Steps

- Step 01: Input image/video utility checks
- Step 02: YOLO `person`/`car` detection
- Step 03: Danger-zone ROI loading and visualization
- Step 04: Single-image risk judgement
- Step 05: Frame-by-frame video risk processing
- Step 06: Temporal stable-risk filtering
- Step 07: Alert capture and CSV logging
- Step 08: Streamlit demo UI
- Step 09: Report and metrics summary generation

## Risk Rule

A `person` detection is considered raw risk when the bottom-center point of its bounding box is inside or on the configured danger-zone polygon.

## Temporal Filter Rule

Stable risk is triggered when at least `{metrics.get("min_risk_count", 0)}` raw-risk frames exist in the latest `{metrics.get("window_size", 0)}` processed frames.

## Alert Logging Rule

Only stable-risk frames are eligible for alert capture. Duplicate alert captures are reduced with a cooldown of `{metrics.get("cooldown_frames", 0)}` frames.

## Key Metrics

| Metric | Value |
| --- | ---: |
| mode | {metrics.get("mode", "unknown")} |
| processed_frames | {metrics.get("processed_frames", 0)} |
| raw_risk_frames | {metrics.get("raw_risk_frames", 0)} |
| stable_risk_frames | {metrics.get("stable_risk_frames", 0)} |
| logged_alert_count | {metrics.get("logged_alert_count", 0)} |
| max_risk_person_count | {metrics.get("max_risk_person_count", 0)} |
| raw_risk_rate | {metrics.get("raw_risk_rate", 0.0):.4f} |
| stable_risk_rate | {metrics.get("stable_risk_rate", 0.0):.4f} |
| alert_log_rate | {metrics.get("alert_log_rate", 0.0):.4f} |

## Generated Result Files

{generated_files_text}

## Limitations

- Fine-tuning is not implemented in the current stage.
- Streamlit is a software demo UI only.
- LED, vehicle control, and other hardware integration are not implemented.
- ROI editing is currently JSON-based.

## Future Improvements

- Add interactive ROI editing.
- Add model evaluation and optional fine-tuning workflow.
- Add operational dashboards for CSV logs and alert history.
- Add hardware integration only after software-side validation is complete.
"""
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def save_metrics_csv(metrics: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ignored_keys = {"generated_files"}
    fieldnames = [key for key in metrics.keys() if key not in ignored_keys]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in fieldnames})

    return str(path)


def create_alert_timeline_plot(log_df, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if log_df is None or log_df.empty:
        ax.text(0.5, 0.5, "No alert log", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        x_column = "timestamp_sec" if "timestamp_sec" in log_df.columns else "frame_index"
        x_values = log_df[x_column].tolist()
        y_values = [1] * len(x_values)
        ax.scatter(x_values, y_values, color="#d62728", s=80)
        ax.vlines(x_values, 0, y_values, color="#d62728", alpha=0.45)
        ax.set_xlabel(x_column)
        ax.set_yticks([1])
        ax.set_yticklabels(["alert"])
        ax.set_title("Alert Timeline")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def create_risk_frame_summary_plot(metrics: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["processed", "raw risk", "stable risk", "logged"]
    values = [
        int(metrics.get("processed_frames", 0)),
        int(metrics.get("raw_risk_frames", 0)),
        int(metrics.get("stable_risk_frames", 0)),
        int(metrics.get("logged_alert_count", 0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#e45756", "#54a24b"])
    ax.set_title("Risk Frame Summary")
    ax.set_ylabel("Frame Count")
    ax.bar_label(bars, labels=[str(value) for value in values], padding=3)
    ax.set_ylim(0, max(values + [1]) * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
