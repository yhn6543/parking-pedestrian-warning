import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report_generator import (
    calculate_rates,
    create_alert_timeline_plot,
    create_risk_frame_summary_plot,
    generate_markdown_report,
    load_csv_log,
    load_json_file,
    save_metrics_csv,
    summarize_alert_summary,
)


DEFAULT_SUMMARY_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "debug" / "streamlit_alert_summary.json",
    PROJECT_ROOT / "outputs" / "debug" / "step07_alert_summary.json",
]
DEFAULT_CSV_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "logs" / "streamlit_risk_log.csv",
    PROJECT_ROOT / "outputs" / "logs" / "step07_risk_log.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 09 report and metrics summary")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def choose_existing_path(explicit_path, candidates):
    if explicit_path:
        return Path(explicit_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def print_failure(error: Exception) -> None:
    print("[STATUS] FAILED")
    print(f"[ERROR] {error}")
    print("[SUGGESTION] Step 07 또는 Step 08을 먼저 실행해 summary JSON과 CSV log를 생성하세요.")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    try:
        summary_path = choose_existing_path(args.summary, DEFAULT_SUMMARY_CANDIDATES)
        if summary_path is None:
            raise FileNotFoundError("No summary JSON found.")

        csv_path = choose_existing_path(args.csv_log, DEFAULT_CSV_CANDIDATES)

        summary = load_json_file(str(summary_path))
        log_df = load_csv_log(str(csv_path)) if csv_path is not None else None

        metrics = summarize_alert_summary(summary)
        metrics.update(calculate_rates(metrics))

        markdown_path = output_dir / "step09_project_summary.md"
        metrics_csv_path = output_dir / "step09_metrics_summary.csv"
        risk_plot_path = figures_dir / "risk_frame_summary.png"
        timeline_plot_path = figures_dir / "alert_timeline.png"

        generated_files = [
            display_path(markdown_path),
            display_path(metrics_csv_path),
            display_path(risk_plot_path),
            display_path(timeline_plot_path),
        ]
        metrics["summary_path"] = display_path(summary_path)
        metrics["csv_log_path"] = display_path(csv_path) if csv_path is not None else "none"
        metrics["generated_files"] = generated_files

        generate_markdown_report(metrics, str(markdown_path))
        save_metrics_csv(metrics, str(metrics_csv_path))
        create_risk_frame_summary_plot(metrics, str(risk_plot_path))
        create_alert_timeline_plot(log_df, str(timeline_plot_path))

        print("[STEP] Step 09 - Report and Metrics Summary")
        print("[INPUT]")
        print(f"* summary: {display_path(summary_path)}")
        print(f"* csv log: {display_path(csv_path) if csv_path is not None else 'none'}")
        print()
        print("[METRICS]")
        print(f"* processed_frames: {metrics['processed_frames']}")
        print(f"* raw_risk_frames: {metrics['raw_risk_frames']}")
        print(f"* stable_risk_frames: {metrics['stable_risk_frames']}")
        print(f"* logged_alert_count: {metrics['logged_alert_count']}")
        print(f"* raw_risk_rate: {metrics['raw_risk_rate']:.4f}")
        print(f"* stable_risk_rate: {metrics['stable_risk_rate']:.4f}")
        print(f"* alert_log_rate: {metrics['alert_log_rate']:.4f}")
        print()
        print("[OUTPUT]")
        print(f"* markdown report: {display_path(markdown_path)}")
        print(f"* metrics csv: {display_path(metrics_csv_path)}")
        print(f"* risk frame summary plot: {display_path(risk_plot_path)}")
        print(f"* alert timeline plot: {display_path(timeline_plot_path)}")
        print()
        print("[STATUS] SUCCESS")
        return 0
    except (FileNotFoundError, ValueError) as error:
        print_failure(error)
        return 1
    except Exception as error:
        print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
