"""
Step 02: Build per-trait SNP effect weights.

For each of 6 dairy traits, designates a subset of SNPs as causal and assigns
effect sizes. Notable genes receive large effects; remaining causal SNPs get
effects from a mixture distribution reflecting polygenic architecture.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.config import (
    NOTABLE_GENES,
    PANEL_PATH,
    RANDOM_SEED,
    TRAITS,
    WEIGHTS_DIR,
)


def _find_notable_indices(panel: list[dict], trait_key: str) -> list[int]:
    """Find panel indices of notable genes associated with a given trait."""
    indices = []
    for gene_key, gene in NOTABLE_GENES.items():
        if trait_key in gene["traits"]:
            for i, snp in enumerate(panel):
                if snp.get("notable") and snp.get("gene") == gene["gene"]:
                    indices.append(i)
                    break
    return indices


def main():
    print("Step 02: Building per-trait SNP effect weights")
    print("-" * 40)

    rng = np.random.default_rng(RANDOM_SEED + 1)

    with open(PANEL_PATH) as f:
        panel = json.load(f)

    n_panel = len(panel)
    all_indices = np.arange(n_panel)

    # Indices of notable/recessive SNPs (excluded from random causal selection)
    special_indices = set()
    for i, snp in enumerate(panel):
        if snp.get("notable") or snp.get("recessive_condition"):
            special_indices.add(i)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for trait_key, trait_info in TRAITS.items():
        n_causal = trait_info["n_causal_snps"]

        # Notable gene indices for this trait (always causal)
        notable_idx = _find_notable_indices(panel, trait_key)

        # Randomly select remaining causal SNPs (excluding special SNPs)
        available = np.array([i for i in all_indices if i not in special_indices])
        n_random = n_causal - len(notable_idx)
        random_idx = rng.choice(available, size=n_random, replace=False).tolist()

        causal_indices = sorted(notable_idx + random_idx)

        # Assign effect sizes
        effects = np.zeros(len(causal_indices))
        for j, idx in enumerate(causal_indices):
            if idx in notable_idx:
                # Large effect for notable genes (~3x SD of other effects)
                effects[j] = rng.normal(0, 0.15)
                if abs(effects[j]) < 0.05:
                    effects[j] = 0.15 * (1 if rng.random() > 0.5 else -1)
            else:
                # Mixture: 80% small, 20% moderate
                if rng.random() < 0.8:
                    effects[j] = rng.normal(0, 0.01)
                else:
                    effects[j] = rng.normal(0, 0.05)

        # Save
        weight_data = {
            "trait": trait_key,
            "n_causal": len(causal_indices),
            "causal_indices": causal_indices,
            "effects": [round(float(e), 6) for e in effects],
            "snp_ids": [panel[i]["id"] for i in causal_indices],
        }

        save_path = WEIGHTS_DIR / f"{trait_key}.json"
        with open(save_path, "w") as f:
            json.dump(weight_data, f)

        notable_effects = [
            effects[j] for j, idx in enumerate(causal_indices) if idx in notable_idx
        ]
        print(f"  {trait_info['name']}:")
        print(f"    Causal SNPs: {len(causal_indices)} ({len(notable_idx)} notable)")
        print(f"    Effect range: [{effects.min():.4f}, {effects.max():.4f}]")
        print(f"    Mean |effect|: {np.abs(effects).mean():.4f}")
        if notable_effects:
            print(f"    Notable gene effects: {[round(e, 4) for e in notable_effects]}")

    print(f"  Saved to: {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
