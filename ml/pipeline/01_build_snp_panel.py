"""
Step 01: Build the Moocleus SNP Panel.

Generates a synthetic but biologically realistic 15,000-SNP panel across
29 bovine autosomes + X chromosome. Notable gene positions and recessive
condition loci are inserted at their real genomic coordinates.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.config import (
    CHR_LENGTHS,
    CHROMOSOMES,
    NOTABLE_GENES,
    N_SNPS,
    PANEL_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    RECESSIVE_CONDITIONS,
)


def main():
    print("Step 01: Building Moocleus SNP Panel")
    print("-" * 40)

    rng = np.random.default_rng(RANDOM_SEED)

    # Collect fixed-position SNPs (notable genes + recessive conditions)
    fixed_snps: dict[tuple, dict] = {}

    for gene_key, gene in NOTABLE_GENES.items():
        chrom = gene["chr"]
        pos = gene["pos"]
        fixed_snps[(chrom, pos)] = {
            "id": gene["rsid"],
            "chr": chrom,
            "pos": pos,
            "gene": gene["gene"],
            "maf": gene["maf"],
            "notable": True,
        }

    for cond_key, cond in RECESSIVE_CONDITIONS.items():
        chrom = cond["chr"]
        # Place condition loci at a distinct position (mid-chromosome offset)
        pos = CHR_LENGTHS[chrom] // 3
        if (chrom, pos) not in fixed_snps:
            fixed_snps[(chrom, pos)] = {
                "id": f"rs_{cond_key}_{chrom}_{pos}",
                "chr": chrom,
                "pos": pos,
                "gene": cond["gene"],
                "maf": cond["carrier_freq"],
                "notable": False,
                "recessive_condition": cond_key,
            }

    # Count fixed SNPs per chromosome
    fixed_per_chr: dict = {}
    for (chrom, _pos), _snp in fixed_snps.items():
        fixed_per_chr[chrom] = fixed_per_chr.get(chrom, 0) + 1

    # Allocate SNPs proportional to chromosome length
    total_length = sum(CHR_LENGTHS[c] for c in CHROMOSOMES)
    n_remaining = N_SNPS - len(fixed_snps)

    alloc: dict = {}
    for chrom in CHROMOSOMES:
        n_fixed = fixed_per_chr.get(chrom, 0)
        n_target = max(0, round(n_remaining * CHR_LENGTHS[chrom] / total_length))
        alloc[chrom] = n_target

    # Adjust to hit exactly n_remaining
    diff = n_remaining - sum(alloc.values())
    sorted_chroms = sorted(CHROMOSOMES, key=lambda c: CHR_LENGTHS[c], reverse=True)
    i = 0
    while diff != 0:
        step = 1 if diff > 0 else -1
        c = sorted_chroms[i % len(sorted_chroms)]
        if alloc[c] + step >= 0:
            alloc[c] += step
            diff -= step
        i += 1

    # Generate SNPs per chromosome
    panel = []

    for chrom in CHROMOSOMES:
        chr_len = CHR_LENGTHS[chrom]
        n_random = alloc[chrom]

        # Get fixed positions for this chromosome
        chr_fixed = {
            pos: snp for (c, pos), snp in fixed_snps.items() if c == chrom
        }
        fixed_positions = set(chr_fixed.keys())

        # Generate random positions avoiding fixed ones
        candidate_positions = rng.integers(1, chr_len, size=n_random * 2)
        candidate_positions = np.unique(candidate_positions)
        candidate_positions = candidate_positions[
            ~np.isin(candidate_positions, list(fixed_positions))
        ]
        if len(candidate_positions) < n_random:
            extra = rng.integers(1, chr_len, size=n_random * 5)
            extra = np.unique(extra)
            extra = extra[~np.isin(extra, list(fixed_positions))]
            candidate_positions = np.unique(
                np.concatenate([candidate_positions, extra])
            )
        selected_positions = np.sort(
            rng.choice(candidate_positions, size=min(n_random, len(candidate_positions)), replace=False)
        )

        # Generate MAFs from Beta distribution (U-shaped, realistic)
        mafs = rng.beta(0.3, 0.3, size=len(selected_positions))
        mafs = np.clip(mafs, 0.01, 0.50)

        # Annotate SNPs near notable genes (within 500kb)
        notable_positions = {
            pos: snp["gene"]
            for pos, snp in chr_fixed.items()
            if snp.get("notable", False)
        }

        for j, pos in enumerate(selected_positions):
            pos = int(pos)
            gene_name = None
            for n_pos, n_gene in notable_positions.items():
                if abs(pos - n_pos) < 500_000:
                    gene_name = n_gene
                    break

            snp = {
                "id": f"rs_{chrom}_{pos}",
                "chr": chrom if isinstance(chrom, str) else int(chrom),
                "pos": pos,
                "maf": round(float(mafs[j]), 4),
            }
            if gene_name:
                snp["gene"] = gene_name
            panel.append(snp)

        # Add fixed SNPs for this chromosome
        for pos, snp in chr_fixed.items():
            entry = {
                "id": snp["id"],
                "chr": snp["chr"] if isinstance(snp["chr"], str) else int(snp["chr"]),
                "pos": int(snp["pos"]),
                "maf": round(float(snp["maf"]), 4),
            }
            if snp.get("gene"):
                entry["gene"] = snp["gene"]
            if snp.get("notable"):
                entry["notable"] = True
            if snp.get("recessive_condition"):
                entry["recessive_condition"] = snp["recessive_condition"]
            panel.append(entry)

    # Sort by chromosome then position
    chr_order = {c: i for i, c in enumerate(CHROMOSOMES)}
    panel.sort(key=lambda s: (chr_order.get(s["chr"], 99), s["pos"]))

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save
    with open(PANEL_PATH, "w") as f:
        json.dump(panel, f)

    # Summary
    print(f"  Total SNPs: {len(panel)}")
    chr_counts = {}
    for snp in panel:
        c = snp["chr"]
        chr_counts[c] = chr_counts.get(c, 0) + 1
    for chrom in CHROMOSOMES:
        print(f"    BTA{chrom}: {chr_counts.get(chrom, 0)} SNPs")

    notable_count = sum(1 for s in panel if s.get("notable"))
    recessive_count = sum(1 for s in panel if s.get("recessive_condition"))
    gene_annotated = sum(1 for s in panel if "gene" in s)
    print(f"  Notable gene SNPs: {notable_count}")
    print(f"  Recessive condition loci: {recessive_count}")
    print(f"  Gene-annotated SNPs: {gene_annotated}")
    print(f"  Saved to: {PANEL_PATH}")


if __name__ == "__main__":
    main()
