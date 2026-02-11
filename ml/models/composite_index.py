"""
Composite Merit Index — Moocleus Score.

Combines GEBVs for all 6 dairy traits into a single 0-100 score,
analogous to USDA Net Merit $. Uses population-standardized trait
values with configurable weights.
"""

import numpy as np
from scipy.stats import norm

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.config import COMPOSITE_WEIGHTS


def compute_moocleus_score(
    gebvs: dict[str, float],
    herd_stats: dict[str, dict[str, float]],
) -> float:
    """
    Compute the Moocleus Score (0-100) from multi-trait GEBVs.

    Args:
        gebvs: dict mapping trait_key -> GEBV value for one animal
        herd_stats: dict mapping trait_key -> {"mean": float, "std": float}

    Returns:
        Moocleus Score on 0-100 scale (percentile of weighted composite)
    """
    # Standardize each GEBV to z-score
    weighted_sum = 0.0
    for trait_key, weight in COMPOSITE_WEIGHTS.items():
        if trait_key not in gebvs:
            continue
        stats = herd_stats.get(trait_key, {"mean": 0, "std": 1})
        std = stats["std"] if stats["std"] > 0 else 1.0
        z = (gebvs[trait_key] - stats["mean"]) / std
        weighted_sum += z * weight

    # Convert to 0-100 percentile using normal CDF
    score = norm.cdf(weighted_sum) * 100
    return round(float(score), 1)


def compute_moocleus_scores_batch(
    gebvs_matrix: dict[str, np.ndarray],
    herd_stats: dict[str, dict[str, float]],
) -> np.ndarray:
    """
    Compute Moocleus Scores for a batch of animals.

    Args:
        gebvs_matrix: dict mapping trait_key -> (n_animals,) array of GEBVs
        herd_stats: dict mapping trait_key -> {"mean": float, "std": float}

    Returns:
        (n_animals,) array of Moocleus Scores (0-100)
    """
    n = None
    weighted_sums = None

    for trait_key, weight in COMPOSITE_WEIGHTS.items():
        if trait_key not in gebvs_matrix:
            continue
        values = gebvs_matrix[trait_key]
        if n is None:
            n = len(values)
            weighted_sums = np.zeros(n)

        stats = herd_stats.get(trait_key, {"mean": 0, "std": 1})
        std = stats["std"] if stats["std"] > 0 else 1.0
        z = (values - stats["mean"]) / std
        weighted_sums += z * weight

    scores = norm.cdf(weighted_sums) * 100
    return np.round(scores, 1)


def assign_badge(score: float) -> str:
    """Assign a display badge based on Moocleus Score."""
    if score >= 75:
        return "Top Pick"
    elif score >= 50:
        return "Above Average"
    else:
        return "Average"


def compute_percentile(value: float, population: np.ndarray) -> int:
    """Compute the percentile rank of a value within a population."""
    return int(round(float(np.mean(population <= value) * 100)))
