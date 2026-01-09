#!/usr/bin/env python3
"""
Generate Phase 3 statistical visualizations.

Creates charts with confidence intervals and error bars
from the aggregated Phase 3 results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.statistical_plots import generate_all_phase3_plots


def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'data' / 'phase3_summary.csv'
    output_dir = project_root / 'visualizations'

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run examples/aggregate_phase3.py first to generate the summary.")
        return

    generate_all_phase3_plots(str(csv_path), str(output_dir))


if __name__ == '__main__':
    main()
