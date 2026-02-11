"""
GBLUP — Genomic Best Linear Unbiased Prediction.

The industry standard for genomic prediction in livestock.
Uses VanRaden (2008) Method 1 for genomic relationship matrix construction.

Reference:
    VanRaden, P.M. (2008). "Efficient methods to compute genomic predictions."
    J. Dairy Sci. 91:4414-4423.
"""

import numpy as np
from scipy import linalg


def compute_grm_vanraden(genotypes: np.ndarray) -> np.ndarray:
    """
    Compute the genomic relationship matrix using VanRaden (2008) Method 1.

    G = ZZ' / (2 * sum(p_i * (1 - p_i)))

    where Z is the centered genotype matrix (z_ij = x_ij - 2*p_j)
    and p_j is the allele frequency at locus j.

    Args:
        genotypes: (n_animals, n_snps) matrix, encoded as 0/1/2

    Returns:
        G: (n_animals, n_animals) genomic relationship matrix
    """
    X = genotypes.astype(np.float64)
    n_animals, n_snps = X.shape

    # Allele frequencies
    p = X.mean(axis=0) / 2.0

    # Center genotypes: Z = X - 2p
    Z = X - 2.0 * p[np.newaxis, :]

    # VanRaden scaling factor
    scale = 2.0 * np.sum(p * (1.0 - p))

    if scale == 0:
        raise ValueError("All SNPs are monomorphic — no genetic variation to analyze.")

    # GRM
    G = Z @ Z.T / scale

    return G


def gblup_solve(
    G: np.ndarray,
    y: np.ndarray,
    h2: float,
    X: np.ndarray | None = None,
) -> dict:
    """
    Solve GBLUP mixed model equations.

    Model: y = Xb + a + e
    where a ~ N(0, G * sigma_a^2) and e ~ N(0, I * sigma_e^2)

    Henderson's mixed model equations:
    [X'X      X'Z    ] [b_hat]   [X'y]
    [Z'X  Z'Z + G_inv*lambda] [a_hat] = [Z'y]

    Simplified when X = 1 (intercept only):
    [G + I*lambda] * a_hat = y - mean(y)

    Args:
        G: (n, n) genomic relationship matrix
        y: (n,) phenotype vector
        h2: heritability estimate
        X: (n, p) fixed effects design matrix (default: intercept only)

    Returns:
        dict with keys: gebvs, intercept, reliability, lambda_val
    """
    n = G.shape[0]
    lambda_val = (1.0 - h2) / h2

    if X is None:
        # Intercept-only model (simplified)
        y_centered = y - np.mean(y)
        lhs = G + np.eye(n) * lambda_val
        gebvs = linalg.solve(lhs, y_centered, assume_a="pos")
        intercept = np.mean(y)
    else:
        # Full mixed model equations
        Z = np.eye(n)
        G_inv = linalg.inv(G + np.eye(n) * 1e-6)  # Regularized inverse

        # Build MME coefficient matrix
        XtX = X.T @ X
        XtZ = X.T @ Z
        ZtX = Z.T @ X
        ZtZ = Z.T @ Z + G_inv * lambda_val

        lhs = np.block([[XtX, XtZ], [ZtX, ZtZ]])
        rhs = np.concatenate([X.T @ y, Z.T @ y])

        solutions = linalg.solve(lhs, rhs)
        p = X.shape[1]
        intercept = solutions[:p]
        gebvs = solutions[p:]

    # Reliability: diagonal of G * (G + I*lambda)^-1
    lhs_inv_diag = np.diag(linalg.inv(G + np.eye(n) * lambda_val))
    reliability = np.diag(G) * (1.0 - lambda_val * lhs_inv_diag)
    reliability = np.clip(reliability, 0, 1)

    return {
        "gebvs": gebvs,
        "intercept": intercept,
        "reliability": reliability,
        "lambda_val": lambda_val,
        "h2": h2,
    }


def predict_gebv(
    G_new_ref: np.ndarray,
    G_ref: np.ndarray,
    gebvs_ref: np.ndarray,
    h2: float,
) -> np.ndarray:
    """
    Predict GEBVs for new animals given their genomic relationship to the reference.

    a_new = G_new_ref @ (G_ref + I*lambda)^-1 @ y_ref

    Args:
        G_new_ref: (n_new, n_ref) genomic relationships between new and reference
        G_ref: (n_ref, n_ref) GRM of reference animals
        gebvs_ref: (n_ref,) estimated breeding values of reference
        h2: heritability

    Returns:
        gebvs_new: (n_new,) predicted GEBVs for new animals
    """
    lambda_val = (1.0 - h2) / h2
    n_ref = G_ref.shape[0]
    lhs = G_ref + np.eye(n_ref) * lambda_val
    weights = linalg.solve(lhs, gebvs_ref)
    return G_new_ref @ weights


def compute_inbreeding(G: np.ndarray) -> np.ndarray:
    """
    Compute inbreeding coefficients from GRM diagonal.

    F_i = G_ii - 1

    Values > 0 indicate inbreeding, < 0 indicates more heterozygosity
    than expected under HWE.
    """
    return np.diag(G) - 1.0
