"""
Cross-validation evaluation and accuracy reporting for all 3 models.

Measures Pearson correlation between predicted GEBVs and true genetic values,
which is the standard accuracy metric in genomic prediction literature.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.bayesian_ridge import train_bayesian_ridge
from ml.models.deep_gblup import train_deep_gblup
from ml.models.gblup import compute_grm_vanraden, gblup_solve
from ml.pipeline.config import RANDOM_SEED, TRAITS


def cross_validate_all(
    genotypes: np.ndarray,
    phenotypes: dict[str, np.ndarray],
    true_genetic_values: dict[str, np.ndarray],
    n_folds: int = 5,
) -> dict:
    """
    Run k-fold CV for GBLUP, Bayesian Ridge, and deepGBLUP.

    Args:
        genotypes: (n_animals, n_snps) matrix
        phenotypes: dict mapping trait_key -> (n_animals,) phenotype array
        true_genetic_values: dict mapping trait_key -> (n_animals,) true genetic values
        n_folds: number of CV folds

    Returns:
        dict: {model_name: {trait_key: pearson_r}}
    """
    n_animals = genotypes.shape[0]
    rng = np.random.default_rng(RANDOM_SEED + 10)
    indices = np.arange(n_animals)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    results = {
        "gblup": {},
        "bayesian_ridge": {},
        "deep_gblup": {},
    }

    for trait_key, trait_info in TRAITS.items():
        y = phenotypes[trait_key]
        true_g = true_genetic_values[trait_key]
        h2 = trait_info["h2"]

        gblup_preds = np.zeros(n_animals)
        bridge_preds = np.zeros(n_animals)

        print(f"  CV for {trait_info['name']}...")

        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

            geno_train = genotypes[train_idx]
            geno_val = genotypes[val_idx]
            y_train = y[train_idx]

            # GBLUP
            G_train = compute_grm_vanraden(geno_train)
            sol = gblup_solve(G_train, y_train, h2=h2)

            # Predict validation: build cross-GRM
            p = geno_train.mean(axis=0) / 2.0
            Z_train = geno_train.astype(np.float64) - 2.0 * p
            Z_val = geno_val.astype(np.float64) - 2.0 * p
            scale = 2.0 * np.sum(p * (1.0 - p))
            G_val_train = Z_val @ Z_train.T / scale if scale > 0 else np.zeros((len(val_idx), len(train_idx)))

            lambda_val = (1 - h2) / h2
            lhs = G_train + np.eye(len(train_idx)) * lambda_val
            weights = np.linalg.solve(lhs, sol["gebvs"])
            gblup_preds[val_idx] = G_val_train @ weights

            # Bayesian Ridge
            br_result = train_bayesian_ridge(geno_train.astype(np.float64), y_train)
            bridge_preds[val_idx] = geno_val.astype(np.float64) @ br_result["coef"] + br_result["intercept"]

        # GBLUP accuracy
        r_gblup, _ = pearsonr(gblup_preds, true_g)
        results["gblup"][trait_key] = round(abs(float(r_gblup)), 4)

        # Bayesian Ridge accuracy
        r_bridge, _ = pearsonr(bridge_preds, true_g)
        results["bayesian_ridge"][trait_key] = round(abs(float(r_bridge)), 4)

        # deepGBLUP (uses its own internal CV)
        G_full = compute_grm_vanraden(genotypes)
        gblup_full = gblup_solve(G_full, y, h2=h2)
        dg_result = train_deep_gblup(
            genotypes=genotypes,
            phenotypes=y,
            gblup_values=gblup_full["gebvs"],
            n_epochs=200,
            patience=20,
            n_folds=n_folds,
            seed=RANDOM_SEED,
        )
        r_dg, _ = pearsonr(dg_result["oof_predictions"], true_g)
        results["deep_gblup"][trait_key] = round(abs(float(r_dg)), 4)

        print(f"    GBLUP r={results['gblup'][trait_key]:.4f}  "
              f"BayesRidge r={results['bayesian_ridge'][trait_key]:.4f}  "
              f"deepGBLUP r={results['deep_gblup'][trait_key]:.4f}")

    return results


def main():
    import pandas as pd
    from ml.pipeline.config import SIMULATED_DIR

    print("Model Evaluation — Cross-Validation")
    print("-" * 40)

    genotypes = np.load(SIMULATED_DIR / "herd_genotypes.npy")
    pheno_df = pd.read_parquet(SIMULATED_DIR / "herd_phenotypes.parquet")
    true_gv = np.load(SIMULATED_DIR / "true_genetic_values.npz")

    phenotypes = {t: pheno_df[t].values for t in TRAITS}
    true_genetic = {t: true_gv[t] for t in TRAITS}

    results = cross_validate_all(genotypes, phenotypes, true_genetic)

    print("\nSummary:")
    print(f"{'Trait':<20} {'GBLUP':>8} {'BayesRidge':>12} {'deepGBLUP':>10}")
    print("-" * 52)
    for trait_key, trait_info in TRAITS.items():
        print(f"{trait_info['name']:<20} "
              f"{results['gblup'][trait_key]:>8.4f} "
              f"{results['bayesian_ridge'][trait_key]:>12.4f} "
              f"{results['deep_gblup'][trait_key]:>10.4f}")


if __name__ == "__main__":
    main()
