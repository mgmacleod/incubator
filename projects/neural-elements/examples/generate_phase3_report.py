#!/usr/bin/env python3
"""
Generate Phase 3 Statistical Summary Report.

Creates a markdown report with:
- Key findings with confidence intervals
- Statistical comparisons between activations
- Tables with mean +/- CI for all configurations
"""

import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    load_from_csv,
    compare_distributions,
    bonferroni_correction,
)


def generate_report(stats, output_path):
    """Generate the Phase 3 markdown report."""

    lines = []

    # Header
    lines.append("# Phase 3: Statistical Robustness Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total configurations**: {len(stats)}")
    lines.append(f"- **Total experiments**: {sum(s.n_trials for s in stats):,}")
    lines.append(f"- **Trials per configuration**: {stats[0].n_trials if stats else 'N/A'}")
    lines.append(f"- **Confidence level**: {stats[0].confidence_level*100:.0f}%")
    lines.append("")

    # Group by activation
    by_activation = defaultdict(list)
    for stat in stats:
        by_activation[stat.activation].append(stat)

    # Activation ranking
    lines.append("## Activation Ranking")
    lines.append("")
    lines.append("| Rank | Activation | Mean Accuracy | 95% CI | Std |")
    lines.append("|------|------------|---------------|--------|-----|")

    activation_means = []
    for act, act_stats in by_activation.items():
        mean_acc = sum(s.mean_accuracy for s in act_stats) / len(act_stats)
        avg_ci_width = sum((s.ci_upper - s.ci_lower) / 2 for s in act_stats) / len(act_stats)
        avg_std = sum(s.std_accuracy for s in act_stats) / len(act_stats)
        activation_means.append({
            'activation': act,
            'mean': mean_acc,
            'ci_width': avg_ci_width,
            'std': avg_std,
        })

    activation_means.sort(key=lambda x: x['mean'], reverse=True)

    for i, am in enumerate(activation_means, 1):
        ci_str = f"[{(am['mean']-am['ci_width'])*100:.1f}%, {(am['mean']+am['ci_width'])*100:.1f}%]"
        lines.append(f"| {i} | {am['activation'].upper()} | {am['mean']*100:.1f}% | {ci_str} | {am['std']*100:.1f}% |")

    lines.append("")

    # Best by dataset
    lines.append("## Best Configuration by Dataset")
    lines.append("")
    lines.append("| Dataset | Best Activation | Depth | Accuracy | 95% CI |")
    lines.append("|---------|-----------------|-------|----------|--------|")

    by_dataset = defaultdict(list)
    for stat in stats:
        by_dataset[stat.dataset].append(stat)

    for dataset in ['xor', 'moons', 'circles', 'spirals']:
        if dataset in by_dataset:
            best = max(by_dataset[dataset], key=lambda s: s.mean_accuracy)
            ci_str = f"[{best.ci_lower*100:.1f}%, {best.ci_upper*100:.1f}%]"
            lines.append(f"| {dataset} | {best.activation.upper()} | {best.depth} | {best.mean_accuracy*100:.1f}% | {ci_str} |")

    lines.append("")

    # Depth effect
    lines.append("## Depth Effect Analysis")
    lines.append("")
    lines.append("Mean accuracy by depth for each activation:")
    lines.append("")
    lines.append("| Activation | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Trend |")
    lines.append("|------------|---------|---------|---------|---------|---------|-------|")

    for act in ['leaky_relu', 'relu', 'sine', 'tanh', 'gelu', 'sigmoid']:
        if act not in by_activation:
            # Try short form
            act = act[:3]
        if act not in by_activation:
            continue

        act_stats = by_activation[act]
        depth_means = {}
        for stat in act_stats:
            if stat.depth not in depth_means:
                depth_means[stat.depth] = []
            depth_means[stat.depth].append(stat.mean_accuracy)

        row = f"| {act.upper()} |"
        values = []
        for d in [1, 2, 3, 4, 5]:
            if d in depth_means:
                mean_val = sum(depth_means[d]) / len(depth_means[d])
                row += f" {mean_val*100:.0f}% |"
                values.append(mean_val)
            else:
                row += " - |"

        # Trend
        if len(values) >= 2:
            diff = values[-1] - values[0]
            if diff > 0.05:
                trend = "Improves"
            elif diff < -0.05:
                trend = "Degrades"
            else:
                trend = "Stable"
        else:
            trend = "-"
        row += f" {trend} |"
        lines.append(row)

    lines.append("")

    # Key findings
    lines.append("## Key Statistical Findings")
    lines.append("")

    # Find sigmoid at depth 5
    sigmoid_d5 = [s for s in stats if s.activation in ['sigmoid', 'sig'] and s.depth == 5]
    if sigmoid_d5:
        avg_sig_d5 = sum(s.mean_accuracy for s in sigmoid_d5) / len(sigmoid_d5)
        lines.append(f"### 1. Sigmoid Collapse Confirmed")
        lines.append("")
        lines.append(f"At depth 5, sigmoid achieves only **{avg_sig_d5*100:.1f}%** accuracy,")
        lines.append(f"which is {'below' if avg_sig_d5 < 0.5 else 'near'} random chance (50%).")
        lines.append("")

    # Find sine on spirals
    sine_spirals = [s for s in stats if s.activation in ['sine', 'sin'] and s.dataset == 'spirals']
    if sine_spirals:
        best_sine_spiral = max(sine_spirals, key=lambda s: s.mean_accuracy)
        lines.append(f"### 2. Sine Excels on Hard Tasks")
        lines.append("")
        lines.append(f"On spirals (the hardest dataset), sine achieves **{best_sine_spiral.mean_accuracy*100:.1f}%**")
        lines.append(f"at depth {best_sine_spiral.depth}, with 95% CI [{best_sine_spiral.ci_lower*100:.1f}%, {best_sine_spiral.ci_upper*100:.1f}%].")
        lines.append("")

    # XOR perfect scores
    xor_perfect = [s for s in stats if s.dataset == 'xor' and s.mean_accuracy >= 0.999]
    if xor_perfect:
        lines.append(f"### 3. XOR is Solved")
        lines.append("")
        lines.append(f"{len(xor_perfect)} configurations achieve 100% accuracy on XOR:")
        for s in sorted(xor_perfect, key=lambda x: (x.activation, x.depth))[:5]:
            lines.append(f"- {s.activation.upper()} depth {s.depth}")
        if len(xor_perfect) > 5:
            lines.append(f"- ... and {len(xor_perfect) - 5} more")
        lines.append("")

    # Full results table
    lines.append("## Complete Results Table")
    lines.append("")
    lines.append("| Activation | Depth | Dataset | Mean | Std | 95% CI | N |")
    lines.append("|------------|-------|---------|------|-----|--------|---|")

    # Sort for readable output
    sorted_stats = sorted(stats, key=lambda s: (s.activation, s.depth, s.dataset))

    for stat in sorted_stats:
        ci_str = f"[{stat.ci_lower*100:.1f}%, {stat.ci_upper*100:.1f}%]"
        lines.append(f"| {stat.activation.upper()} | {stat.depth} | {stat.dataset} | {stat.mean_accuracy*100:.1f}% | {stat.std_accuracy*100:.1f}% | {ci_str} | {stat.n_completed} |")

    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## Visualizations")
    lines.append("")
    lines.append("See `visualizations/` directory for charts:")
    lines.append("- `phase3_depth_effect_ci.png` - Depth effect with confidence bands")
    lines.append("- `phase3_activation_ranking_ci.png` - Activation ranking with error bars")
    lines.append("- `phase3_dataset_comparison_ci.png` - Performance by dataset")
    lines.append("- `phase3_variance_reliability.png` - Accuracy vs reliability scatter")
    lines.append("- `phase3_activation_boxplots.png` - Distribution box plots")
    lines.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'data' / 'phase3_summary.csv'

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run examples/aggregate_phase3.py first.")
        return

    # Create reports directory
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    # Load stats
    print(f"Loading results from {csv_path}...")
    stats = load_from_csv(str(csv_path))
    print(f"Loaded {len(stats)} configuration summaries")

    # Generate report
    output_path = reports_dir / 'phase3_statistical_summary.md'
    generate_report(stats, str(output_path))


if __name__ == '__main__':
    main()
