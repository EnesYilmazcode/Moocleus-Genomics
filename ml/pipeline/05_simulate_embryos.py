"""
Step 05: Simulate embryo cohorts.

Selects 5 sire-dam pairs from the herd and produces 10 embryos per mating
via Mendelian segregation. Each parent transmits one allele per locus
with probability determined by their genotype.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.config import (
    EMBRYO_NAMES,
    EMBRYOS_PER_MATING,
    N_HERD,
    N_MATINGS,
    RANDOM_SEED,
    SIMULATED_DIR,
    TRAITS,
)


def main():
    print("Step 05: Simulating embryo cohorts")
    print("-" * 40)

    rng = np.random.default_rng(RANDOM_SEED + 4)

    herd_geno = np.load(SIMULATED_DIR / "herd_genotypes.npy")
    true_gv = np.load(SIMULATED_DIR / "true_genetic_values.npz")
    n_snps = herd_geno.shape[1]

    # Compute mean standardized genetic value per animal for selecting parents
    trait_keys = list(TRAITS.keys())
    standardized = []
    for t in trait_keys:
        vals = true_gv[t]
        std = vals.std()
        if std > 0:
            standardized.append((vals - vals.mean()) / std)
        else:
            standardized.append(np.zeros_like(vals))
    mean_gv = np.mean(standardized, axis=0)

    # Select sires: top N_MATINGS animals by mean genetic value
    sire_indices = np.argsort(mean_gv)[-N_MATINGS:][::-1]

    # Select dams: random from top 50% (excluding sires)
    top_half = set(np.argsort(mean_gv)[N_HERD // 2:])
    available_dams = sorted(top_half - set(sire_indices))
    dam_indices = rng.choice(available_dams, size=N_MATINGS, replace=False)

    embryo_genotypes = []
    embryo_metadata = []
    name_idx = 0

    for mating_idx in range(N_MATINGS):
        sire_geno = herd_geno[sire_indices[mating_idx]]
        dam_geno = herd_geno[dam_indices[mating_idx]]

        sire_id = f"COW-{sire_indices[mating_idx]:03d}"
        dam_id = f"COW-{dam_indices[mating_idx]:03d}"

        print(f"  Mating {mating_idx + 1}: {sire_id} x {dam_id}")

        for _emb in range(EMBRYOS_PER_MATING):
            # Mendelian segregation: each parent transmits one allele
            # genotype 0 (AA) -> always transmits A (0)
            # genotype 1 (Aa) -> 50% chance A or a
            # genotype 2 (aa) -> always transmits a (1)
            sire_allele = (rng.random(n_snps) < (sire_geno / 2.0)).astype(np.int8)
            dam_allele = (rng.random(n_snps) < (dam_geno / 2.0)).astype(np.int8)
            embryo_geno = sire_allele + dam_allele

            embryo_genotypes.append(embryo_geno)
            embryo_metadata.append({
                "embryo_id": f"EMB-{name_idx + 1:03d}",
                "name": EMBRYO_NAMES[name_idx % len(EMBRYO_NAMES)],
                "sire_id": sire_id,
                "dam_id": dam_id,
                "mating_group": mating_idx + 1,
            })
            name_idx += 1

    embryo_geno_array = np.array(embryo_genotypes, dtype=np.int8)
    np.save(SIMULATED_DIR / "embryo_genotypes.npy", embryo_geno_array)

    embryo_df = pd.DataFrame(embryo_metadata)
    embryo_df.to_parquet(SIMULATED_DIR / "embryo_metadata.parquet", index=False)

    print(f"  Total embryos: {len(embryo_metadata)}")
    print(f"  Genotype shape: {embryo_geno_array.shape}")
    print(f"  Saved to: {SIMULATED_DIR}")


if __name__ == "__main__":
    main()
