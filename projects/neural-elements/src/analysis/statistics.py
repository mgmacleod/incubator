"""
Statistical analysis utilities for Neural Elements experiments.

Provides functions for:
- Confidence interval calculation (parametric and bootstrap)
- Distribution comparison (t-test, Mann-Whitney U)
- Multiple comparison correction (Bonferroni)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import numpy as np


@dataclass
class ConfidenceInterval:
    """Result of a confidence interval calculation."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n: int
    confidence: float


@dataclass
class ComparisonResult:
    """Result of comparing two distributions."""
    statistic: float
    p_value: float
    test_name: str
    significant: bool  # At alpha=0.05
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> ConfidenceInterval:
    """
    Compute a confidence interval using the t-distribution.

    Assumes approximately normal distribution of the sample mean
    (valid for n >= 20 by CLT, reasonable for n >= 5).

    Args:
        values: List of sample values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval with mean, bounds, std, and sample size
    """
    n = len(values)
    if n == 0:
        return ConfidenceInterval(
            mean=float('nan'),
            ci_lower=float('nan'),
            ci_upper=float('nan'),
            std=float('nan'),
            n=0,
            confidence=confidence
        )

    if n == 1:
        val = values[0]
        return ConfidenceInterval(
            mean=val,
            ci_lower=val,
            ci_upper=val,
            std=0.0,
            n=1,
            confidence=confidence
        )

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # Sample std with Bessel's correction
    se = std / math.sqrt(n)

    # t critical value for two-tailed test
    # Using approximation for t-distribution
    # For n >= 30, t approaches z
    alpha = 1 - confidence
    if n >= 30:
        # Use z-value approximation
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        t_crit = z
    else:
        # Use t-distribution critical values (precomputed for common cases)
        # t_crit for two-tailed 95% CI
        t_table_95 = {
            2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
            7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
            12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
            17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
            22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
            27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045
        }
        t_crit = t_table_95.get(n, 2.0)  # Default to ~2.0 for moderate n

    margin = t_crit * se

    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        std=std,
        n=n,
        confidence=confidence
    )


def compute_bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None
) -> ConfidenceInterval:
    """
    Compute a confidence interval using bootstrap resampling.

    Non-parametric method that doesn't assume normality.
    Useful when sample size is small or distribution is skewed.

    Args:
        values: List of sample values
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        ConfidenceInterval with mean, bounds, std, and sample size
    """
    n = len(values)
    if n == 0:
        return ConfidenceInterval(
            mean=float('nan'),
            ci_lower=float('nan'),
            ci_upper=float('nan'),
            std=float('nan'),
            n=0,
            confidence=confidence
        )

    if n == 1:
        val = values[0]
        return ConfidenceInterval(
            mean=val,
            ci_lower=val,
            ci_upper=val,
            std=0.0,
            n=1,
            confidence=confidence
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std=std,
        n=n,
        confidence=confidence
    )


def compare_distributions(
    group_a: List[float],
    group_b: List[float],
    test: str = 'auto'
) -> ComparisonResult:
    """
    Compare two distributions for significant difference.

    Args:
        group_a: First group of values
        group_b: Second group of values
        test: Test to use ('t', 'mannwhitney', or 'auto')
              'auto' uses t-test if both groups have n >= 20, else Mann-Whitney

    Returns:
        ComparisonResult with test statistic, p-value, and significance
    """
    a = np.array(group_a)
    b = np.array(group_b)

    n_a, n_b = len(a), len(b)

    if n_a < 2 or n_b < 2:
        return ComparisonResult(
            statistic=float('nan'),
            p_value=1.0,
            test_name='insufficient_data',
            significant=False
        )

    # Choose test
    if test == 'auto':
        test = 't' if (n_a >= 20 and n_b >= 20) else 'mannwhitney'

    if test == 't':
        # Welch's t-test (doesn't assume equal variance)
        result = _welch_t_test(a, b)
        # Cohen's d effect size
        pooled_std = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled_std > 0:
            effect_size = (np.mean(a) - np.mean(b)) / pooled_std
        else:
            effect_size = 0.0

        return ComparisonResult(
            statistic=result['t'],
            p_value=result['p'],
            test_name='welch_t',
            significant=result['p'] < 0.05,
            effect_size=effect_size,
            effect_size_name='cohens_d'
        )

    elif test == 'mannwhitney':
        # Mann-Whitney U test (non-parametric)
        result = _mann_whitney_u(a, b)

        # Rank-biserial correlation as effect size
        n_total = n_a * n_b
        if n_total > 0:
            effect_size = 1 - (2 * result['U']) / n_total
        else:
            effect_size = 0.0

        return ComparisonResult(
            statistic=result['U'],
            p_value=result['p'],
            test_name='mann_whitney_u',
            significant=result['p'] < 0.05,
            effect_size=effect_size,
            effect_size_name='rank_biserial'
        )

    else:
        raise ValueError(f"Unknown test: {test}")


def _welch_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Perform Welch's t-test (unequal variance t-test).

    Implements the formula without scipy dependency.
    """
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    # Welch's t-statistic
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return {'t': 0.0, 'p': 1.0, 'df': n_a + n_b - 2}

    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else n_a + n_b - 2

    # Approximate p-value using normal distribution for large df
    # For more accuracy, would need scipy.stats.t
    p = 2 * (1 - _normal_cdf(abs(t)))

    return {'t': float(t), 'p': float(p), 'df': float(df)}


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test.

    Implements the formula without scipy dependency.
    """
    n_a, n_b = len(a), len(b)

    # Combine and rank
    combined = np.concatenate([a, b])
    ranks = _rank_data(combined)

    # Sum of ranks for group a
    R_a = np.sum(ranks[:n_a])

    # U statistics
    U_a = R_a - n_a * (n_a + 1) / 2
    U_b = n_a * n_b - U_a
    U = min(U_a, U_b)

    # Normal approximation for p-value (valid for n >= 20)
    mean_U = n_a * n_b / 2
    # Correction for ties would go here
    std_U = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    if std_U == 0:
        return {'U': float(U), 'p': 1.0}

    z = (U - mean_U) / std_U
    p = 2 * (1 - _normal_cdf(abs(z)))

    return {'U': float(U), 'p': float(p), 'z': float(z)}


def _rank_data(data: np.ndarray) -> np.ndarray:
    """Rank data, handling ties with average ranks."""
    n = len(data)
    sorted_indices = np.argsort(data)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        # Find all elements with the same value (ties)
        while j < n - 1 and data[sorted_indices[j]] == data[sorted_indices[j + 1]]:
            j += 1
        # Average rank for ties
        avg_rank = (i + j) / 2 + 1  # +1 because ranks start at 1
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1

    return ranks


def _normal_cdf(x: float) -> float:
    """
    Approximate cumulative distribution function for standard normal.

    Uses Abramowitz and Stegun approximation (error < 7.5e-8).
    """
    # Constants for approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Handle negative x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)

    return 0.5 * (1.0 + sign * y)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired family-wise error rate

    Returns:
        List of (adjusted_p, significant) tuples
    """
    n = len(p_values)
    if n == 0:
        return []

    adjusted_alpha = alpha / n

    return [
        (min(p * n, 1.0), p < adjusted_alpha)
        for p in p_values
    ]


def aggregate_trial_statistics(
    trial_accuracies: Dict[str, List[float]],
    confidence: float = 0.95
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate statistics across trials for multiple configurations.

    Args:
        trial_accuracies: Dict mapping config_key to list of accuracy values
        confidence: Confidence level for intervals

    Returns:
        Dict mapping config_key to statistical summary
    """
    results = {}

    for config_key, accuracies in trial_accuracies.items():
        ci = compute_confidence_interval(accuracies, confidence)
        results[config_key] = {
            'mean': ci.mean,
            'std': ci.std,
            'ci_lower': ci.ci_lower,
            'ci_upper': ci.ci_upper,
            'n_trials': ci.n,
            'min': min(accuracies) if accuracies else float('nan'),
            'max': max(accuracies) if accuracies else float('nan'),
            'confidence': confidence,
        }

    return results


@dataclass
class PairwiseComparison:
    """Result of comparing two groups."""
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    difference: float  # mean_a - mean_b
    p_value: float
    p_value_adjusted: float
    significant: bool  # After correction
    effect_size: float
    test_name: str


def pairwise_activation_comparison(
    activation_accuracies: Dict[str, List[float]],
    alpha: float = 0.05
) -> List[PairwiseComparison]:
    """
    Perform pairwise comparisons between all activation functions.

    Uses Bonferroni correction for multiple comparisons.

    Args:
        activation_accuracies: Dict mapping activation name to list of accuracies
        alpha: Significance level

    Returns:
        List of PairwiseComparison results, sorted by p-value
    """
    activations = list(activation_accuracies.keys())
    n_comparisons = len(activations) * (len(activations) - 1) // 2

    comparisons = []
    p_values = []

    # All pairwise comparisons
    for i, act_a in enumerate(activations):
        for act_b in activations[i + 1:]:
            vals_a = activation_accuracies[act_a]
            vals_b = activation_accuracies[act_b]

            result = compare_distributions(vals_a, vals_b)

            comparison = PairwiseComparison(
                group_a=act_a,
                group_b=act_b,
                mean_a=float(np.mean(vals_a)),
                mean_b=float(np.mean(vals_b)),
                difference=float(np.mean(vals_a) - np.mean(vals_b)),
                p_value=result.p_value,
                p_value_adjusted=0.0,  # Will be filled in
                significant=False,  # Will be filled in
                effect_size=result.effect_size if result.effect_size else 0.0,
                test_name=result.test_name,
            )
            comparisons.append(comparison)
            p_values.append(result.p_value)

    # Apply Bonferroni correction
    corrected = bonferroni_correction(p_values, alpha)

    for comp, (adj_p, sig) in zip(comparisons, corrected):
        comp.p_value_adjusted = adj_p
        comp.significant = sig

    # Sort by p-value
    comparisons.sort(key=lambda c: c.p_value)

    return comparisons


def rank_activations_with_significance(
    activation_accuracies: Dict[str, List[float]],
    alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Rank activations and group by statistical equivalence.

    Activations in the same group are not significantly different from each other.

    Args:
        activation_accuracies: Dict mapping activation name to list of accuracies
        alpha: Significance level

    Returns:
        List of dicts with activation, mean, rank, and significance_group
    """
    # Get pairwise comparisons
    comparisons = pairwise_activation_comparison(activation_accuracies, alpha)

    # Build significance matrix
    activations = list(activation_accuracies.keys())
    sig_matrix: Dict[Tuple[str, str], bool] = {}

    for comp in comparisons:
        sig_matrix[(comp.group_a, comp.group_b)] = comp.significant
        sig_matrix[(comp.group_b, comp.group_a)] = comp.significant

    # Rank by mean accuracy
    ranked = []
    for act, vals in activation_accuracies.items():
        ranked.append({
            'activation': act,
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'n': len(vals),
        })

    ranked.sort(key=lambda x: x['mean'], reverse=True)

    # Assign ranks
    for i, item in enumerate(ranked):
        item['rank'] = i + 1

    # Assign significance groups using connected components
    # Two activations are in same group if NOT significantly different
    groups: Dict[str, int] = {}
    current_group = 0

    for i, item in enumerate(ranked):
        act = item['activation']
        if act not in groups:
            groups[act] = current_group
            # Find all activations not significantly different from this one
            for j, other_item in enumerate(ranked):
                other_act = other_item['activation']
                if other_act not in groups:
                    # Check if significantly different
                    key = (act, other_act) if act < other_act else (other_act, act)
                    if key in sig_matrix and not sig_matrix[key]:
                        groups[other_act] = current_group
            current_group += 1

    for item in ranked:
        item['significance_group'] = groups[item['activation']]

    return ranked
