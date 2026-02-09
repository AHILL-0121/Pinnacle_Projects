#!/usr/bin/env python3
"""
CLI entry point for the CrewAI Logistics Optimization System.

Usage:
    python main.py                          # run with sample data
    python main.py --data path/to/data.json # run with custom data
    python main.py --quiet                  # suppress agent chatter
    python main.py --output report.md       # save output to file

SRS FR-1 through FR-5 are satisfied through this pipeline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so local imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.schemas import LogisticsData
from crew import build_crew
from config import OUTPUT_DIR


SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_logistics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrewAI Logistics Optimization System — MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py
  python main.py --data data/sample_logistics.json
  python main.py --quiet --output output/report.md
        """,
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=str(SAMPLE_DATA_PATH),
        help="Path to logistics data JSON file (default: sample data)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save final report to this file path",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose agent output",
    )
    return parser.parse_args()


def load_data(path: str) -> LogisticsData:
    """Load and validate logistics data from a JSON file."""
    abs_path = Path(path).resolve()
    if not abs_path.exists():
        print(f"[ERROR] Data file not found: {abs_path}")
        sys.exit(1)

    print(f"[INFO] Loading logistics data from {abs_path}")
    data = LogisticsData.from_json_file(str(abs_path))
    print(
        f"[INFO] Loaded {len(data.products)} products, "
        f"{len(data.routes)} routes, {len(data.inventory)} inventory records"
    )
    return data


def save_output(content: str, path: str) -> None:
    """Persist the crew output to a file."""
    abs_path = Path(path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Report saved to {abs_path}")


def main() -> None:
    args = parse_args()

    # Override verbosity if --quiet
    if args.quiet:
        os.environ["CREW_VERBOSE"] = "false"
        # Re-import config to pick up change
        import config
        config.VERBOSE = False

    print("=" * 60)
    print("  CrewAI Logistics Optimization System — MVP")
    print("=" * 60)

    # ── Step 1: Load data (FR-1) ─────────────────────────────────────
    logistics_data = load_data(args.data)

    # ── Step 2: Build & run crew (FR-2 → FR-5) ──────────────────────
    print("\n[INFO] Building crew…")
    crew = build_crew(logistics_data)

    print("[INFO] Kicking off analysis pipeline…\n")
    start = time.time()
    result = crew.kickoff()
    elapsed = time.time() - start

    # ── Step 3: Output results ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OPTIMIZATION REPORT")
    print("=" * 60)

    report_text = str(result)
    print(report_text)

    print(f"\n[INFO] Completed in {elapsed:.1f}s")

    # Save to file if requested
    if args.output:
        save_output(report_text, args.output)
    else:
        # Default: save to output dir with timestamp
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(OUTPUT_DIR, f"report_{ts}.md")
        save_output(report_text, default_path)


if __name__ == "__main__":
    main()
