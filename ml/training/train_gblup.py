"""
Train GBLUP on simulated herd data and save solutions.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.gblup import compute_grm_vanraden, gblup_solve
from ml.pipeline.config import ARTIFACTS_DIR, SIMULATED_DIR, TRAITS


def main():
    print("Training GBLUP")
    print("-" * 40)

    genotypes = np.load(SIMULATED_DIR / "herd_genotypes.npy")
    phenotypes = pd.read_parquet(SIMULATED_DIR / "herd_phenotypes.parquet")

    # Compute GRM once (shared across all traits)
    print("  Computing genomic relationship matrix...")
    G = compute_grm_vanraden(genotypes)
    print(f"  GRM shape: {G.shape}, mean diagonal: {np.diag(G).mean():.4f}")

    solutions = {"grm": G}

    for trait_key, trait_info in TRAITS.items():
        y = phenotypes[trait_key].values
        result = gblup_solve(G, y, h2=trait_info["h2"])

        solutions[trait_key] = {
            "gebvs": result["gebvs"],
            "intercept": result["intercept"],
            "reliability": result["reliability"],
            "h2": result["h2"],
        }

        print(f"  {trait_info['name']}:")
        print(f"    GEBV range: [{result['gebvs'].min():.3f}, {result['gebvs'].max():.3f}]")
        print(f"    Mean reliability: {result['reliability'].mean():.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "gblup_solutions.pkl", "wb") as f:
        pickle.dump(solutions, f)

    print(f"  Saved to: {ARTIFACTS_DIR / 'gblup_solutions.pkl'}")


if __name__ == "__main__":
    main()
