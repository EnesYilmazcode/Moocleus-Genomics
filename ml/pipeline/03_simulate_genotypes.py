"""
Step 03: Simulate herd genotypes.

Generates genotype matrices for 200 herd animals using realistic allele
frequencies from the SNP panel and block-based linkage disequilibrium.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.config import N_HERD, PANEL_PATH, RANDOM_SEED, SIMULATED_DIR


def main():
    print("Step 03: Simulating herd genotypes")
    print("-" * 40)

    rng = np.random.default_rng(RANDOM_SEED + 2)

    with open(PANEL_PATH) as f:
        panel = json.load(f)

    n_snps = len(panel)
    mafs = np.array([snp["maf"] for snp in panel])

    genotypes = np.zeros((N_HERD, n_snps), dtype=np.int8)

    # Process in blocks for LD structure
    block_size = 20

    for start in range(0, n_snps, block_size):
        end = min(start + block_size, n_snps)
        k = end - start
        block_mafs = mafs[start:end]

        # Build correlation matrix with exponential decay
        idx = np.arange(k)
        corr = np.exp(-0.3 * np.abs(idx[:, None] - idx[None, :]))

        # Ensure positive definiteness
        corr += np.eye(k) * 1e-6

        # Sample multivariate normal
        z = rng.multivariate_normal(np.zeros(k), corr, size=N_HERD)

        # Convert to genotypes using MAF-based thresholds
        for j in range(k):
            p = block_mafs[j]
            # Thresholds for 0/1/2 encoding based on allele frequency
            # P(geno=0) = (1-p)^2, P(geno=1) = 2p(1-p), P(geno=2) = p^2
            t1 = norm.ppf((1 - p) ** 2)
            t2 = norm.ppf(1 - p**2)
            genotypes[:, start + j] = np.digitize(z[:, j], [t1, t2]).astype(np.int8)

    # Ensure output directory
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SIMULATED_DIR / "herd_genotypes.npy", genotypes)

    # Summary statistics
    geno_counts = np.array([(genotypes == v).sum() for v in [0, 1, 2]])
    total = genotypes.size
    realized_maf = genotypes.mean(axis=0) / 2.0

    print(f"  Shape: {genotypes.shape}")
    print(f"  Genotype frequencies: 0={geno_counts[0]/total:.3f}, 1={geno_counts[1]/total:.3f}, 2={geno_counts[2]/total:.3f}")
    print(f"  Mean MAF: {realized_maf.mean():.3f} (target: {mafs.mean():.3f})")
    print(f"  MAF range: [{realized_maf.min():.3f}, {realized_maf.max():.3f}]")

    # Sample LD between adjacent SNPs
    ld_samples = []
    for i in range(min(100, n_snps - 1)):
        r = np.corrcoef(genotypes[:, i].astype(float), genotypes[:, i + 1].astype(float))[0, 1]
        ld_samples.append(abs(r))
    print(f"  Mean |r| adjacent SNPs (first 100): {np.mean(ld_samples):.3f}")
    print(f"  Saved to: {SIMULATED_DIR / 'herd_genotypes.npy'}")


if __name__ == "__main__":
    main()
