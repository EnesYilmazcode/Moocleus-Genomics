# Moocleus Genomics

**Breed Better. Know More.**

A polygenic selection engine for dairy cattle embryos. Analyzes 15,000+ genetic markers to predict production, health, and longevity traits using GBLUP and deep learning hybrids.

This is a bovine analog of [Nucleus Genomics](https://mynucleus.com/) — a direct-to-consumer genomics company that calculates polygenic risk scores for humans and offers embryo screening. Moocleus does the same thing, but for dairy cattle — where genomic prediction is actually *ahead* of human genomics in accuracy.

## Features

- **6 dairy traits** — Milk Yield, Fat %, Protein %, Fertility, Somatic Cell Score, Longevity
- **3 ML models** — GBLUP (industry standard), Bayesian Ridge (SNP-level effects), deepGBLUP (Conv1d + GBLUP hybrid)
- **50 simulated embryos** from 5 elite matings, ranked by composite Moocleus Score (0-100)
- **Interactive Manhattan plot** across all 29 bovine autosomes + X chromosome
- **Embryo comparison** — radar charts, bell curves, notable gene annotations, carrier status
- **6-page Streamlit app** with Nucleus-inspired design

## Tech Stack

| Layer | Technology |
|---|---|
| App | Python + Streamlit |
| Visualization | Plotly |
| ML (statistical) | NumPy, SciPy — GBLUP with VanRaden GRM |
| ML (Bayesian) | scikit-learn — BayesianRidge for SNP effects |
| ML (deep learning) | PyTorch — deepGBLUP (locally-connected CNN + FC + GBLUP) |
| Data | pandas, pyarrow |

## Quick Start

Demo data is included — the app works immediately after install:

```bash
git clone https://github.com/EnesYilmazcode/Moocleus-Genomics.git
cd Moocleus-Genomics
pip install -e .
python -m streamlit run app/Home.py
```

Then open http://localhost:8501 in your browser.

## Regenerating Demo Data

To regenerate the simulated data and retrain all models from scratch (~5 minutes):

```bash
python -m ml.pipeline.run_all
```

This runs the full pipeline:
1. **Build SNP panel** — 15,000 markers with real chromosomal architecture
2. **Build effect weights** — per-trait causal SNP effects (notable genes like DGAT1, ABCG2 get large effects)
3. **Simulate genotypes** — 200-cow herd with linkage disequilibrium
4. **Simulate phenotypes** — 6 traits from published heritabilities
5. **Simulate embryos** — 50 embryos via Mendelian segregation from elite matings
6. **Export** — trains GBLUP, Bayesian Ridge, and deepGBLUP, then packages everything into `demo_data.json`

## Project Structure

```
moocleus-genomics/
├── app/                          # Streamlit application
│   ├── Home.py                   # Landing page
│   ├── pages/
│   │   ├── 1_Dashboard.py        # Herd overview + Manhattan plot
│   │   ├── 2_Embryo_Selection.py # Embryo grid + comparison
│   │   ├── 3_Embryo_Report.py    # Individual embryo detail
│   │   ├── 4_Model_Performance.py# Model accuracy comparison
│   │   └── 5_About.py            # Methodology + citations
│   ├── components/
│   │   ├── charts.py             # Plotly chart factory functions
│   │   ├── embryo_card.py        # Embryo display card
│   │   └── metrics.py            # Metric card components
│   └── styles/
│       └── theme.css             # Nucleus-inspired CSS theme
│
├── ml/                           # ML pipeline
│   ├── models/
│   │   ├── gblup.py              # VanRaden GRM + mixed model equations
│   │   ├── bayesian_ridge.py     # scikit-learn BayesianRidge wrapper
│   │   ├── deep_gblup.py         # PyTorch deepGBLUP (Conv1d + FC + GBLUP)
│   │   └── composite_index.py    # Moocleus Score computation
│   ├── pipeline/
│   │   ├── config.py             # All constants, traits, genes, parameters
│   │   ├── 01_build_snp_panel.py
│   │   ├── 02_build_effect_weights.py
│   │   ├── 03_simulate_genotypes.py
│   │   ├── 04_simulate_phenotypes.py
│   │   ├── 05_simulate_embryos.py
│   │   ├── 06_export_demo_data.py
│   │   └── run_all.py            # Pipeline orchestrator
│   ├── training/
│   │   ├── train_gblup.py
│   │   ├── train_deep_gblup.py
│   │   └── evaluate.py           # Cross-validation for all models
│   └── data/processed/
│       └── demo_data.json        # Pre-computed app data (included in repo)
│
├── pyproject.toml
├── Dockerfile
└── CLAUDE.md                     # Detailed project blueprint
```

## How It Works

### GBLUP (Genomic Best Linear Unbiased Prediction)
The industry gold standard. Constructs a genomic relationship matrix (GRM) using VanRaden (2008) Method 1, then solves Henderson's mixed model equations to estimate breeding values.

### Bayesian Ridge Regression
Estimates individual SNP effects, enabling per-marker polygenic scores and Manhattan plot visualization. Uses automatic relevance determination for regularization.

### deepGBLUP (Deep Learning Hybrid)
Combines a locally-connected Conv1d pathway (learns haplotype patterns from groups of 50 adjacent SNPs) with pre-computed GBLUP values through a learnable alpha parameter. Based on Lee et al. (2023).

### Moocleus Score
A composite merit index combining all 6 trait GEBVs with weights approximating USDA Net Merit $: production (65%), fertility (15%), health (10%), longevity (10%). Scaled 0-100 within each embryo cohort.

## Docker

```bash
docker build -t moocleus .
docker run -p 8501:8501 moocleus
```

## References

1. VanRaden, P.M. (2008). "Efficient methods to compute genomic predictions." *J. Dairy Sci.* 91:4414-4423.
2. Lee, J. et al. (2023). "deepGBLUP: Joint deep learning and GBLUP framework for genomic prediction." *Genetics Selection Evolution* 55:25.
3. Wiggans, G.R. et al. (2017). "Genomic selection in dairy cattle: the USDA experience." *Ann. Rev. Anim. Biosci.* 5:309-327.
4. Hu, Z.L. et al. (2019). "Animal QTLdb: an improved database tool for livestock animal QTL/association data." *Nucleic Acids Res.* 47:D694-D698.
