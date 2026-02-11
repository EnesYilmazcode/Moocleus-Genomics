"""
Bayesian Ridge Regression for SNP-effect estimation.

Provides individual SNP effect weights needed for:
1. Computing polygenic scores for new embryos
2. Manhattan plot data (effect size per SNP)
3. Feature importance / top-SNP identification
"""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge


def train_bayesian_ridge(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    max_iter: int = 500,
) -> dict:
    """
    Train BayesianRidge regression on genotype matrix.

    Args:
        genotypes: (n_animals, n_snps) matrix, 0/1/2 encoded
        phenotypes: (n_animals,) trait values
        max_iter: maximum iterations for convergence

    Returns:
        dict with coef, intercept, predictions, and alpha/lambda params
    """
    model = BayesianRidge(max_iter=max_iter, tol=1e-4)
    model.fit(genotypes, phenotypes)

    return {
        "model": model,
        "coef": model.coef_,
        "intercept": float(model.intercept_),
        "predictions": model.predict(genotypes),
        "alpha": float(model.alpha_),
        "lambda": float(model.lambda_),
    }


def predict_bayesian_ridge(
    model_result: dict,
    new_genotypes: np.ndarray,
) -> np.ndarray:
    """Predict GEBVs for new animals using trained Bayesian Ridge."""
    return new_genotypes @ model_result["coef"] + model_result["intercept"]


def snp_effects_to_manhattan(
    coef: np.ndarray,
    panel: list[dict],
) -> dict:
    """
    Convert SNP coefficients to Manhattan plot data.

    Computes pseudo -log10(p-values) from effect sizes:
      z = |effect| / MAD(effects), p = 2 * (1 - Phi(z))

    Args:
        coef: (n_snps,) SNP effect estimates
        panel: list of SNP dicts with chr, pos, id, gene fields

    Returns:
        dict with chr, pos, neg_log_p, effect, snp_id, gene lists
    """
    abs_coef = np.abs(coef)
    mad = np.median(abs_coef)
    if mad == 0:
        mad = abs_coef.mean()
    if mad == 0:
        mad = 1e-10

    z_scores = abs_coef / mad
    p_values = 2.0 * norm.sf(z_scores)
    p_values = np.clip(p_values, 1e-300, 1.0)
    neg_log_p = -np.log10(p_values)

    result = {
        "chr": [],
        "pos": [],
        "neg_log_p": [],
        "effect": [],
        "snp_id": [],
        "gene": [],
    }

    for i, snp in enumerate(panel):
        result["chr"].append(snp["chr"])
        result["pos"].append(snp["pos"])
        result["neg_log_p"].append(round(float(neg_log_p[i]), 4))
        result["effect"].append(round(float(coef[i]), 6))
        result["snp_id"].append(snp["id"])
        result["gene"].append(snp.get("gene", ""))

    return result
