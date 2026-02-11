"""
Step 06: Export demo data.

Trains all 3 models, computes embryo GEBVs and Moocleus Scores,
assembles Manhattan plot data, and packages everything into a single
demo_data.json file for the Streamlit app.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.bayesian_ridge import (
    predict_bayesian_ridge,
    snp_effects_to_manhattan,
    train_bayesian_ridge,
)
from ml.models.composite_index import (
    assign_badge,
    compute_moocleus_score,
    compute_percentile,
)
from ml.models.deep_gblup import predict_deep_gblup, train_deep_gblup
from ml.models.gblup import compute_grm_vanraden, compute_inbreeding, gblup_solve, predict_gebv
from ml.pipeline.config import (
    ARTIFACTS_DIR,
    NOTABLE_GENES,
    PANEL_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    RECESSIVE_CONDITIONS,
    SIMULATED_DIR,
    TRAITS,
    COMPOSITE_WEIGHTS,
)


def _subsample_manhattan(manhattan_data: dict, max_points: int = 5000) -> dict:
    """Subsample Manhattan data, keeping significant SNPs and sampling the rest."""
    n = len(manhattan_data["chr"])
    neg_log_p = np.array(manhattan_data["neg_log_p"])

    # Keep all SNPs with -log10(p) > 2 or with gene annotations
    significant = neg_log_p > 2
    annotated = np.array([bool(g) for g in manhattan_data["gene"]])
    keep = significant | annotated

    n_keep = keep.sum()
    n_remaining = max(0, max_points - n_keep)

    if n_remaining > 0 and (~keep).sum() > 0:
        rng = np.random.default_rng(42)
        remaining_idx = np.where(~keep)[0]
        sample_idx = rng.choice(remaining_idx, size=min(n_remaining, len(remaining_idx)), replace=False)
        keep[sample_idx] = True

    result = {}
    keep_indices = np.where(keep)[0]
    for key in manhattan_data:
        result[key] = [manhattan_data[key][i] for i in keep_indices]

    return result


def main():
    print("Step 06: Exporting demo data")
    print("-" * 40)

    # Load all simulated data
    with open(PANEL_PATH) as f:
        panel = json.load(f)

    herd_geno = np.load(SIMULATED_DIR / "herd_genotypes.npy")
    herd_pheno = pd.read_parquet(SIMULATED_DIR / "herd_phenotypes.parquet")
    embryo_geno = np.load(SIMULATED_DIR / "embryo_genotypes.npy")
    embryo_meta = pd.read_parquet(SIMULATED_DIR / "embryo_metadata.parquet")
    true_gv = np.load(SIMULATED_DIR / "true_genetic_values.npz")

    n_herd = herd_geno.shape[0]
    n_embryos = embryo_geno.shape[0]

    # ── TRAIN MODELS ──────────────────────────────────────────────
    print("  Training models...")

    # Compute GRM for herd
    G_herd = compute_grm_vanraden(herd_geno)
    inbreeding_herd = compute_inbreeding(G_herd)

    # Build combined GRM for embryo prediction
    all_geno = np.vstack([herd_geno, embryo_geno])
    G_all = compute_grm_vanraden(all_geno)
    G_embryo_herd = G_all[n_herd:, :n_herd]  # Cross-GRM
    G_embryo = G_all[n_herd:, n_herd:]  # Embryo GRM
    inbreeding_embryo = compute_inbreeding(G_embryo)

    herd_gebvs = {"gblup": {}, "bayesian_ridge": {}, "deep_gblup": {}}
    embryo_gebvs = {"gblup": {}, "bayesian_ridge": {}, "deep_gblup": {}}
    manhattan_data = {}
    model_performance = {"gblup": {}, "bayesian_ridge": {}, "deep_gblup": {}}

    for trait_key, trait_info in TRAITS.items():
        print(f"    {trait_info['name']}...")
        y = herd_pheno[trait_key].values
        true_g = true_gv[trait_key]
        h2 = trait_info["h2"]

        # GBLUP
        gblup_result = gblup_solve(G_herd, y, h2=h2)
        herd_gebvs["gblup"][trait_key] = gblup_result["gebvs"].tolist()
        embryo_gblup = predict_gebv(G_embryo_herd, G_herd, gblup_result["gebvs"], h2)
        embryo_gebvs["gblup"][trait_key] = embryo_gblup.tolist()

        # Bayesian Ridge
        br_result = train_bayesian_ridge(herd_geno.astype(np.float64), y)
        herd_gebvs["bayesian_ridge"][trait_key] = (br_result["predictions"] - br_result["intercept"]).tolist()
        embryo_br = predict_bayesian_ridge(br_result, embryo_geno.astype(np.float64)) - br_result["intercept"]
        embryo_gebvs["bayesian_ridge"][trait_key] = embryo_br.tolist()

        # Manhattan plot from Bayesian Ridge coefficients
        full_manhattan = snp_effects_to_manhattan(br_result["coef"], panel)
        manhattan_data[trait_key] = _subsample_manhattan(full_manhattan)

        # deepGBLUP
        dg_result = train_deep_gblup(
            genotypes=herd_geno,
            phenotypes=y,
            gblup_values=gblup_result["gebvs"],
            n_epochs=200,
            patience=20,
            n_folds=5,
            seed=RANDOM_SEED,
        )
        # In-sample predictions from final model (for herd GEBVs and display)
        herd_dg = predict_deep_gblup(dg_result, herd_geno, gblup_result["gebvs"])
        herd_gebvs["deep_gblup"][trait_key] = herd_dg.tolist()
        embryo_dg = predict_deep_gblup(dg_result, embryo_geno, embryo_gblup)
        embryo_gebvs["deep_gblup"][trait_key] = embryo_dg.tolist()

        # Model accuracy (all in-sample for fair comparison in demo)
        from scipy.stats import pearsonr
        r_gblup, _ = pearsonr(gblup_result["gebvs"], true_g)
        r_br, _ = pearsonr(br_result["predictions"] - br_result["intercept"], true_g)
        r_dg, _ = pearsonr(herd_dg, true_g)
        model_performance["gblup"][trait_key] = round(abs(float(r_gblup)), 4)
        model_performance["bayesian_ridge"][trait_key] = round(abs(float(r_br)), 4)
        model_performance["deep_gblup"][trait_key] = round(abs(float(r_dg)), 4)

    # ── COMPUTE HERD STATISTICS ───────────────────────────────────
    print("  Computing herd statistics...")

    herd_stats = {}
    for trait_key in TRAITS:
        # Use Bayesian Ridge GEBVs for stats (direct SNP effects, more variable)
        vals = np.array(herd_gebvs["bayesian_ridge"][trait_key])
        herd_stats[trait_key] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }

    # ── BUILD EMBRYO RECORDS ──────────────────────────────────────
    print("  Building embryo records...")

    # Build panel index for notable genes and recessive conditions
    notable_panel_idx = {}
    recessive_panel_idx = {}
    for i, snp in enumerate(panel):
        if snp.get("notable"):
            notable_panel_idx[snp["gene"]] = i
        if snp.get("recessive_condition"):
            recessive_panel_idx[snp["recessive_condition"]] = i

    embryos = []
    for emb_i in range(n_embryos):
        meta = embryo_meta.iloc[emb_i]

        # GEBVs — use Bayesian Ridge (direct SNP effects, good variance)
        gebvs = {}
        percentiles = {}
        for trait_key in TRAITS:
            val = embryo_gebvs["bayesian_ridge"][trait_key][emb_i]
            gebvs[trait_key] = round(float(val), 4)
            herd_vals = np.array(herd_gebvs["bayesian_ridge"][trait_key])
            percentiles[trait_key] = compute_percentile(val, herd_vals)

        # Moocleus Score
        score = compute_moocleus_score(gebvs, herd_stats)
        badge = assign_badge(score)

        # Notable SNP genotypes
        notable_snps = []
        for gene_key, gene_info in NOTABLE_GENES.items():
            if gene_info["gene"] in notable_panel_idx:
                idx = notable_panel_idx[gene_info["gene"]]
                geno_val = int(embryo_geno[emb_i, idx])
                # Map 0/1/2 to genotype descriptions
                alleles = ["Favorable homozygous", "Heterozygous carrier", "Alternate homozygous"]
                notable_snps.append({
                    "gene": gene_info["gene"],
                    "variant": gene_info["variant"],
                    "genotype": geno_val,
                    "genotype_desc": alleles[geno_val],
                    "effect": gene_info["effect"],
                })

        # Carrier status for recessive conditions
        carrier_status = {}
        for cond_key in RECESSIVE_CONDITIONS:
            if cond_key in recessive_panel_idx:
                idx = recessive_panel_idx[cond_key]
                geno_val = int(embryo_geno[emb_i, idx])
                carrier_status[cond_key] = geno_val >= 1  # 1=carrier, 2=affected
            else:
                carrier_status[cond_key] = False

        embryo_record = {
            "id": meta["embryo_id"],
            "name": meta["name"],
            "sire_id": meta["sire_id"],
            "dam_id": meta["dam_id"],
            "mating_group": int(meta["mating_group"]),
            "gebvs": gebvs,
            "percentiles": percentiles,
            "moocleus_score": score,
            "badge": badge,
            "notable_snps": notable_snps,
            "carrier_status": carrier_status,
            "inbreeding_coeff": round(float(inbreeding_embryo[emb_i]), 4),
        }
        embryos.append(embryo_record)

    # Rescale Moocleus Scores to 0-100 within the embryo cohort
    # This ranks embryos relative to each other (like Nucleus ranks embryos)
    raw_scores = np.array([e["moocleus_score"] for e in embryos])
    min_s, max_s = raw_scores.min(), raw_scores.max()
    score_range = max_s - min_s if max_s > min_s else 1.0
    for emb in embryos:
        # Map to 5-98 range (avoid exact 0 or 100)
        scaled = 5 + 93 * (emb["moocleus_score"] - min_s) / score_range
        emb["moocleus_score"] = round(float(scaled), 1)
        emb["badge"] = assign_badge(emb["moocleus_score"])

    # ── ASSEMBLE FINAL JSON ───────────────────────────────────────
    print("  Assembling demo_data.json...")

    # Phenotype data for herd distributions
    herd_phenotypes = {}
    for trait_key in TRAITS:
        herd_phenotypes[trait_key] = [round(float(v), 2) for v in herd_pheno[trait_key].values]

    demo_data = {
        "panel": {
            "n_snps": len(panel),
            "snps": panel,  # Full panel for reference
        },
        "manhattan": manhattan_data,
        "herd": {
            "n_animals": n_herd,
            "phenotypes": herd_phenotypes,
            "gebvs": {
                model: {
                    trait: [round(float(v), 4) for v in vals]
                    for trait, vals in traits.items()
                }
                for model, traits in herd_gebvs.items()
            },
            "stats": herd_stats,
        },
        "embryos": embryos,
        "model_performance": model_performance,
        "trait_config": {
            k: {
                "name": v["name"],
                "unit": v["unit"],
                "h2": v["h2"],
                "mean": v["mean"],
                "std": v["std"],
                "description": v["description"],
            }
            for k, v in TRAITS.items()
        },
        "composite_weights": dict(COMPOSITE_WEIGHTS),
    }

    output_path = PROCESSED_DIR / "demo_data.json"
    with open(output_path, "w") as f:
        json.dump(demo_data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Output: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Embryos: {len(embryos)}")
    print(f"  Model performance:")
    for model, traits in model_performance.items():
        avg_r = np.mean(list(traits.values()))
        print(f"    {model}: avg r={avg_r:.4f}")

    # Also save model artifacts for reference
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "snp_effects_all_traits.json", "w") as f:
        effects_data = {}
        for trait_key in TRAITS:
            trait_manhattan = manhattan_data[trait_key]
            effects_data[trait_key] = {
                "snp_ids": trait_manhattan["snp_id"],
                "effects": trait_manhattan["effect"],
            }
        json.dump(effects_data, f)


if __name__ == "__main__":
    main()
