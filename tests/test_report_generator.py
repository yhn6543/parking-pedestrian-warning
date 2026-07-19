import csv

import pytest

from src.report_generator import (
    calculate_rates,
    generate_markdown_report,
    load_json_file,
    save_metrics_csv,
    summarize_alert_summary,
)


def test_summarize_alert_summary_extracts_main_values() -> None:
    summary = {
        "mode": "MOCK_PERSON",
        "processed_frames": 30,
        "raw_risk_frames": 30,
        "stable_risk_frames": 28,
        "logged_alert_count": 3,
        "max_risk_person_count": 1,
        "window_size": 5,
        "min_risk_count": 3,
        "cooldown_frames": 10,
    }

    metrics = summarize_alert_summary(summary)

    assert metrics["mode"] == "MOCK_PERSON"
    assert metrics["processed_frames"] == 30
    assert metrics["logged_alert_count"] == 3


def test_calculate_rates_uses_processed_frames() -> None:
    metrics = {
        "processed_frames": 10,
        "raw_risk_frames": 5,
        "stable_risk_frames": 3,
        "logged_alert_count": 1,
    }

    rates = calculate_rates(metrics)

    assert rates["raw_risk_rate"] == 0.5
    assert rates["stable_risk_rate"] == 0.3
    assert rates["alert_log_rate"] == 0.1


def test_calculate_rates_handles_zero_processed_frames() -> None:
    rates = calculate_rates({"processed_frames": 0})

    assert rates["raw_risk_rate"] == 0.0
    assert rates["stable_risk_rate"] == 0.0
    assert rates["alert_log_rate"] == 0.0


def test_generate_markdown_report_creates_file(tmp_path) -> None:
    output_path = tmp_path / "report.md"
    metrics = {
        "mode": "MOCK_PERSON",
        "processed_frames": 5,
        "raw_risk_frames": 5,
        "stable_risk_frames": 3,
        "logged_alert_count": 1,
        "max_risk_person_count": 1,
        "window_size": 5,
        "min_risk_count": 3,
        "cooldown_frames": 10,
        "raw_risk_rate": 1.0,
        "stable_risk_rate": 0.6,
        "alert_log_rate": 0.2,
    }

    saved_path = generate_markdown_report(metrics, str(output_path))

    assert saved_path == str(output_path)
    assert "Parking Pedestrian Warning" in output_path.read_text(encoding="utf-8")


def test_save_metrics_csv_creates_file(tmp_path) -> None:
    output_path = tmp_path / "metrics.csv"
    metrics = {"processed_frames": 5, "raw_risk_rate": 1.0}

    saved_path = save_metrics_csv(metrics, str(output_path))

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert saved_path == str(output_path)
    assert rows[0]["processed_frames"] == "5"


def test_load_json_file_raises_value_error_for_invalid_json(tmp_path) -> None:
    json_path = tmp_path / "broken.json"
    json_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError):
        load_json_file(str(json_path))
