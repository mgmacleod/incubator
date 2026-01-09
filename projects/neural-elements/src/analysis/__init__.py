"""Analysis module for Neural Elements experiments.

Provides tools for:
- Configuration selection and experiment matrix generation
- Statistical analysis (confidence intervals, significance tests)
- Result aggregation across trials
"""

from .config_selection import Phase3Config, Phase4Config, Phase5Config, generate_phase3_configs
from .statistics import (
    compute_confidence_interval,
    compute_bootstrap_ci,
    compare_distributions,
    bonferroni_correction,
)
from .aggregator import (
    ConfigurationStats,
    aggregate_experiments,
    export_to_csv,
    load_from_csv,
    get_summary_tables,
    print_summary,
)

__all__ = [
    # Config selection
    'Phase3Config',
    'Phase4Config',
    'Phase5Config',
    'generate_phase3_configs',
    # Statistics
    'compute_confidence_interval',
    'compute_bootstrap_ci',
    'compare_distributions',
    'bonferroni_correction',
    # Aggregation
    'ConfigurationStats',
    'aggregate_experiments',
    'export_to_csv',
    'load_from_csv',
    'get_summary_tables',
    'print_summary',
]
