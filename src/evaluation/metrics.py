"""
Statistical evaluation metrics for the A/B test.

Stratum 1 (verifiable): two-proportion z-test at alpha=0.05.
Stratum 2 (open-ended): win-rate with explicit bias caveats, no significance claims.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StratumResult:
    stratum: str
    n_baseline: int
    n_treatment: int
    correct_baseline: int
    correct_treatment: int
    accuracy_baseline: float
    accuracy_treatment: float
    delta: float
    z_stat: float
    p_value: float
    significant: bool
    alpha: float
    confidence_interval_95: tuple


@dataclass
class OpenEndedResult:
    n_examples: int
    baseline_win_rate: float
    treatment_win_rate: float
    tie_rate: float
    bias_caveats: List[str] = field(default_factory=list)
    note: str = "Directional signal only. Not used for significance claims."


def two_proportion_z_test(
    n_baseline: int,
    n_treatment: int,
    correct_baseline: int,
    correct_treatment: int,
    alpha: float = 0.05,
) -> Dict:
    """
    Two-proportion z-test for equality of proportions.

    H0: p_treatment == p_baseline
    H1: p_treatment != p_baseline (two-tailed)

    Uses pooled proportion estimate under H0.

    Args:
        n_baseline: Total baseline examples.
        n_treatment: Total treatment examples.
        correct_baseline: Correct baseline predictions.
        correct_treatment: Correct treatment predictions.
        alpha: Significance level.

    Returns:
        Dict with z_stat, p_value, significant, ci_95.
    """
    p_base = correct_baseline / n_baseline
    p_treat = correct_treatment / n_treatment
    p_pool = (correct_baseline + correct_treatment) / (n_baseline + n_treatment)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_baseline + 1 / n_treatment))

    if se == 0:
        logger.warning("Zero standard error — both proportions are identical.")
        return {
            "z_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "ci_95": (p_treat - p_base, p_treat - p_base),
        }

    z_stat = (p_treat - p_base) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    se_diff = np.sqrt(
        p_base * (1 - p_base) / n_baseline
        + p_treat * (1 - p_treat) / n_treatment
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    delta = p_treat - p_base
    ci_95 = (delta - z_crit * se_diff, delta + z_crit * se_diff)

    return {
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "ci_95": (float(ci_95[0]), float(ci_95[1])),
    }


def compute_stratum_result(
    stratum_name: str,
    baseline_correct: List[bool],
    treatment_correct: List[bool],
    alpha: float = 0.05,
) -> StratumResult:
    """
    Compute StratumResult for a single verifiable stratum.

    Args:
        stratum_name: Human-readable stratum label.
        baseline_correct: Per-example binary correctness for baseline.
        treatment_correct: Per-example binary correctness for treatment.
        alpha: Significance level.

    Returns:
        StratumResult with all statistics populated.
    """
    assert len(baseline_correct) == len(treatment_correct), (
        f"Stratum {stratum_name}: baseline and treatment must have equal n. "
        f"Got {len(baseline_correct)} vs {len(treatment_correct)}."
    )

    n = len(baseline_correct)
    n_correct_base = sum(baseline_correct)
    n_correct_treat = sum(treatment_correct)
    acc_base = n_correct_base / n
    acc_treat = n_correct_treat / n
    delta = acc_treat - acc_base

    test = two_proportion_z_test(n, n, n_correct_base, n_correct_treat, alpha)

    return StratumResult(
        stratum=stratum_name,
        n_baseline=n,
        n_treatment=n,
        correct_baseline=n_correct_base,
        correct_treatment=n_correct_treat,
        accuracy_baseline=acc_base,
        accuracy_treatment=acc_treat,
        delta=delta,
        z_stat=test["z_stat"],
        p_value=test["p_value"],
        significant=test["significant"],
        alpha=alpha,
        confidence_interval_95=test["ci_95"],
    )


def compute_open_ended_result(
    baseline_wins: int,
    treatment_wins: int,
    ties: int,
    bias_caveats: List[str],
) -> OpenEndedResult:
    """
    Summarize open-ended judging results without significance claims.

    Args:
        baseline_wins: Examples where baseline was judged better.
        treatment_wins: Examples where treatment was judged better.
        ties: Examples judged as ties.
        bias_caveats: List of known biases to surface in the report.

    Returns:
        OpenEndedResult with win rates and caveats.
    """
    total = baseline_wins + treatment_wins + ties
    return OpenEndedResult(
        n_examples=total,
        baseline_win_rate=baseline_wins / total,
        treatment_win_rate=treatment_wins / total,
        tie_rate=ties / total,
        bias_caveats=bias_caveats,
    )


def format_results_table(
    stratum_results: List[StratumResult],
    open_ended_result: Optional[OpenEndedResult] = None,
) -> str:
    """Format evaluation results as a readable summary string."""
    lines = [
        "",
        "=" * 72,
        "STRATIFIED A/B EVALUATION RESULTS",
        "=" * 72,
        "",
        "Stratum 1: Verifiable Tasks (primary claim)",
        "-" * 72,
        f"{'Stratum':<24} {'Base':>8} {'Treat':>8} {'Delta':>8} {'p-value':>10} {'Sig?':>6}",
        "-" * 72,
    ]

    for r in stratum_results:
        sig_marker = "YES *" if r.significant else "no"
        lines.append(
            f"{r.stratum:<24} "
            f"{r.accuracy_baseline:>8.3f} "
            f"{r.accuracy_treatment:>8.3f} "
            f"{r.delta:>+8.3f} "
            f"{r.p_value:>10.4f} "
            f"{sig_marker:>6}"
        )
        lines.append(
            f"{'':>24} n={r.n_baseline} | "
            f"95% CI: [{r.confidence_interval_95[0]:+.3f}, {r.confidence_interval_95[1]:+.3f}]"
        )

    if open_ended_result is not None:
        lines += [
            "",
            "Stratum 2: Open-Ended Tasks (directional only)",
            "-" * 72,
            f"  Treatment win rate : {open_ended_result.treatment_win_rate:.3f}",
            f"  Baseline win rate  : {open_ended_result.baseline_win_rate:.3f}",
            f"  Tie rate           : {open_ended_result.tie_rate:.3f}",
            f"  n                  : {open_ended_result.n_examples}",
            "",
            "  NOTE: " + open_ended_result.note,
            "  Bias caveats:",
        ]
        for caveat in open_ended_result.bias_caveats:
            lines.append(f"    - {caveat}")

    lines += ["", "=" * 72, ""]
    return "\n".join(lines)
