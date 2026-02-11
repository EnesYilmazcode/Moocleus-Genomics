"""
Step 04: Simulate trait phenotypes.

Generates phenotypes for all 6 dairy traits using causal SNP architecture
from Step 02 and genotypes from Step 03. Phenotypes follow the infinitesimal
model: phenotype = genetic_value + environmental_noise.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.config import N_HERD, RANDOM_SEED, SIMULATED_DIR, TRAITS, WEIGHTS_DIR


def main():
    print("Step 04: Simulating trait phenotypes")
    print("-" * 40)

    rng = np.random.default_rng(RANDOM_SEED + 3)

    genotypes = np.load(SIMULATED_DIR / "herd_genotypes.npy")

    phenotype_data = {"animal_id": [f"COW-{i:03d}" for i in range(N_HERD)]}
    true_genetic = {}

    for trait_key, trait_info in TRAITS.items():
        with open(WEIGHTS_DIR / f"{trait_key}.json") as f:
            weights = json.load(f)

        causal_idx = np.array(weights["causal_indices"])
        effects = np.array(weights["effects"])

        # Compute genetic values
        g = genotypes[:, causal_idx].astype(np.float64) @ effects

        # Scale genetic variance to match h2 * phenotypic variance
        h2 = trait_info["h2"]
        target_var_g = h2 * trait_info["std"] ** 2

        g_std = np.std(g)
        if g_std > 0:
            g = g * np.sqrt(target_var_g) / g_std

        # Center genetic values
        g = g - np.mean(g)

        # Environmental noise
        var_e = target_var_g * (1 - h2) / h2
        noise = rng.normal(0, np.sqrt(var_e), size=N_HERD)

        # Phenotype = genetic + environmental + population mean
        phenotype = g + noise + trait_info["mean"]

        phenotype_data[trait_key] = phenotype
        true_genetic[trait_key] = g

        # Realized heritability check
        realized_h2 = np.var(g) / np.var(phenotype - trait_info["mean"])

        print(f"  {trait_info['name']}:")
        print(f"    Mean: {phenotype.mean():.2f} (target: {trait_info['mean']})")
        print(f"    SD: {phenotype.std():.2f} (target: {trait_info['std']})")
        print(f"    h2: {realized_h2:.3f} (target: {h2})")

    # Save phenotypes
    df = pd.DataFrame(phenotype_data)
    df.to_parquet(SIMULATED_DIR / "herd_phenotypes.parquet", index=False)

    # Save true genetic values for model evaluation
    np.savez(SIMULATED_DIR / "true_genetic_values.npz", **true_genetic)

    print(f"  Saved phenotypes to: {SIMULATED_DIR / 'herd_phenotypes.parquet'}")
    print(f"  Saved true genetic values to: {SIMULATED_DIR / 'true_genetic_values.npz'}")


if __name__ == "__main__":
    main()
