"""
Train deepGBLUP on simulated herd data and save model weights.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.deep_gblup import train_deep_gblup
from ml.pipeline.config import ARTIFACTS_DIR, RANDOM_SEED, SIMULATED_DIR, TRAITS


def main():
    print("Training deepGBLUP")
    print("-" * 40)

    genotypes = np.load(SIMULATED_DIR / "herd_genotypes.npy")
    phenotypes = pd.read_parquet(SIMULATED_DIR / "herd_phenotypes.parquet")

    # Load pre-computed GBLUP solutions
    with open(ARTIFACTS_DIR / "gblup_solutions.pkl", "rb") as f:
        gblup_solutions = pickle.load(f)

    all_results = {}

    for trait_key, trait_info in TRAITS.items():
        print(f"  Training for {trait_info['name']}...")

        y = phenotypes[trait_key].values
        gblup_gebvs = gblup_solutions[trait_key]["gebvs"]

        result = train_deep_gblup(
            genotypes=genotypes,
            phenotypes=y,
            gblup_values=gblup_gebvs,
            n_epochs=200,
            lr=1e-3,
            patience=20,
            n_folds=5,
            seed=RANDOM_SEED,
        )

        all_results[trait_key] = {
            "model_state_dict": result["model_state_dict"],
            "n_snps": result["n_snps"],
            "final_alpha": result["final_alpha"],
            "cv_losses": result["cv_losses"],
        }

        print(f"    Alpha (GBLUP weight): {1 - result['final_alpha']:.3f}")
        print(f"    Mean CV loss: {np.mean(result['cv_losses']):.4f}")

    torch.save(all_results, ARTIFACTS_DIR / "deep_gblup_weights.pt")
    print(f"  Saved to: {ARTIFACTS_DIR / 'deep_gblup_weights.pt'}")


if __name__ == "__main__":
    main()
