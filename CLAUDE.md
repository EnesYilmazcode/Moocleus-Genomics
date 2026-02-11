# CLAUDE.md — Moocleus Genomics Project Blueprint

> **Moocleus Genomics** — "Breed Better. Know More."
>
> A polygenic selection engine for dairy cattle embryos. Analyzes 15,000+ genetic markers
> to predict production, health, and longevity traits using GBLUP and deep learning hybrids.

---

## Why This Project Exists

This project is a bovine analog of [Nucleus Genomics](https://mynucleus.com/) — a direct-to-consumer
genomics company that calculates polygenic risk scores (PRS) for humans and offers embryo screening
via their "Origin" models. Moocleus does the same thing, but for dairy cattle.

The twist: bovine genomics is actually *ahead* of human genomics in genomic prediction.
Cattle have larger genotyped training populations (6.6M+ Holsteins), higher prediction
accuracies for production traits, and no ethical constraints on selection. This means Moocleus
can train real models on real data and achieve genuinely meaningful prediction accuracies —
it is not a toy project.

**Core parallel to Nucleus:**
| Nucleus Genomics (Humans) | Moocleus Genomics (Cattle) |
|---|---|
| Polygenic Risk Scores for 800+ diseases | Genomic Estimated Breeding Values for 6 dairy traits |
| Origin embryo screening models | Embryo selection + ranking engine |
| 7M genetic markers analyzed | 15K informative markers (Moocleus Panel) |
| Deep learning + statistical hybrids | deepGBLUP + GBLUP hybrid |
| Consumer web dashboard | Streamlit app with Plotly visualizations |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **App** | Python + Streamlit | Multi-page web application with custom CSS theming |
| **Visualization** | Plotly | Manhattan plots, radar charts, trait distributions, bell curves |
| **ML (statistical)** | NumPy, SciPy | GBLUP — genomic relationship matrix + mixed model equations |
| **ML (Bayesian)** | scikit-learn | BayesianRidge — SNP-effect estimation for PRS and feature importance |
| **ML (deep learning)** | PyTorch | deepGBLUP — locally-connected CNN + FC hybrid combined with GBLUP |
| **Data** | pandas, pyarrow | Data wrangling, parquet storage for processed artifacts |
| **Package mgmt** | UV | Fast Python dependency management |
| **Deployment** | Streamlit Cloud / Docker | Free hosting or containerized deployment |

---

## Datasets

All datasets are publicly available. No restricted or proprietary data is required.

### Primary: Animal QTLdb (Trait-SNP Associations)

- **URL:** https://www.animalgenome.org/cgi-bin/QTLdb/index
- **Contents:** 5,920+ cattle QTL/association data points with trait-SNP mappings
- **Format:** Tab-delimited with columns: trait, chromosome, position, p-value, effect size, breed
- **Use:** Provides the SNP effect sizes needed for polygenic score calculation and the causal architecture for realistic phenotype simulation

### Reference: Illumina BovineHD BeadChip Manifest

- **URL:** https://www.illumina.com/products/by-type/microarray-kits/bovinehd.html
- **Contents:** 777,962 SNP positions across all 29 autosomes + X chromosome
- **Use:** Defines the "chip" — real SNP positions used to build the Moocleus Panel. Ensures Manhattan plots show real chromosomal architecture

### Allele Frequencies: Bovine Genome Variation Database (BGVD)

- **URL:** http://animal.omics.pro/code/index.php/BosVar
- **Contents:** ~60.44M SNPs with minor allele frequencies across 54 cattle breeds
- **Use:** Provides population allele frequencies for centering genotypes in GRM computation and for generating realistic simulated genotypes

### Gene Annotations: Bovine Genome Database (BGD)

- **URL:** https://bovinegenome.elsiklab.missouri.edu/
- **Contents:** ARS-UCD2.0 assembly — latest bovine reference genome with gene annotations, QTL data, RNA-seq tracks
- **Use:** SNP-to-gene mapping for biological annotation, BANN model connectivity, and labeling Manhattan plot peaks

### Extended Reference: 1000 Bull Genomes Project

- **URL:** NCBI Project ID PRJEB42783 (https://db.cngb.org/ for Run 9 data)
- **Contents:** 2,703 whole-genome sequences from diverse cattle breeds, 84M SNPs
- **Use:** Reference panel for imputation validation and cross-breed variant calling

---

## ML Models

### Tier 1: GBLUP — Genomic Best Linear Unbiased Prediction

The industry gold standard. Every national cattle breeding program uses this. Including it demonstrates understanding of the foundational method.

**Mathematical formulation:**

```
y = Xb + Za + e

where:
  y = phenotypic observations (or deregressed proofs)
  X = design matrix for fixed effects (herd, year, season)
  b = fixed effects solutions
  Z = design matrix for random animal effects
  a = breeding values ~ N(0, G * sigma_a^2)
  G = genomic relationship matrix (VanRaden 2008, Method 1)
  e = residuals ~ N(0, I * sigma_e^2)
```

**GRM construction (VanRaden Method 1):**

```python
def compute_grm_vanraden(genotypes: np.ndarray) -> np.ndarray:
    """
    genotypes: (n_animals, n_snps) matrix, encoded as 0/1/2
    Returns: (n_animals, n_animals) genomic relationship matrix
    """
    p = genotypes.mean(axis=0) / 2.0          # allele frequencies
    Z = genotypes - 2.0 * p[np.newaxis, :]    # center genotypes
    scale = 2.0 * np.sum(p * (1.0 - p))       # VanRaden scaling
    G = Z @ Z.T / scale
    return G
```

**GBLUP prediction (simplified mixed model equations):**

```python
def gblup_predict(G: np.ndarray, y: np.ndarray, h2: float) -> np.ndarray:
    """
    lambda_val = sigma_e^2 / sigma_a^2 = (1 - h2) / h2
    Solves: [G + I*lambda] * a_hat = y
    Returns: Genomic Estimated Breeding Values (GEBVs)
    """
    lambda_val = (1.0 - h2) / h2
    n = G.shape[0]
    lhs = G + np.eye(n) * lambda_val
    a_hat = np.linalg.solve(lhs, y)
    return a_hat
```

**Published heritabilities for variance component estimation:**

| Trait | h2 | lambda | Notes |
|---|---|---|---|
| Milk yield | 0.28 | 2.57 | High accuracy, large effect QTLs |
| Fat % | 0.26 | 2.85 | Strong DGAT1 effect on BTA14 |
| Protein % | 0.23 | 3.35 | |
| Fertility (calving interval) | 0.04 | 24.0 | Low h2, hardest to predict |
| SCS (somatic cell score / health) | 0.12 | 7.33 | |
| Longevity | 0.05 | 19.0 | Low h2 |

### Tier 2: Bayesian Ridge — SNP-Effect Model

Provides individual SNP effect weights, which are needed for:
1. Computing polygenic scores for new embryos: `score = genotype_vector @ effect_weights`
2. Manhattan plot data (effect size per SNP per chromosome)
3. Feature importance / top-SNP identification

```python
from sklearn.linear_model import BayesianRidge

model = BayesianRidge(max_iter=500, tol=1e-4)
model.fit(genotypes, phenotypes)
# model.coef_ -> SNP effect estimates (n_snps,)
```

### Tier 3: deepGBLUP — Deep Learning Hybrid (The Impressive One)

This mirrors what Nucleus does with their Origin models — combining deep learning with statistical genomics.

**Architecture:**

```
Input: SNP genotypes (n_snps,) as 0/1/2
    |
    v
[Locally-Connected Conv1d]
    Groups of 50 adjacent SNPs -> learns local haplotype patterns
    Conv1d(1, 4, kernel_size=50, stride=50) -> ReLU
    |
    v
[Fully-Connected Pathway]
    FC(local_features, 512) -> BatchNorm -> ReLU -> Dropout(0.3)
    FC(512, 128) -> BatchNorm -> ReLU -> Dropout(0.2)
    FC(128, 1) -> g_dl (deep learning genomic value)
    |
    |    [Simultaneously]
    |
[GBLUP Branch]
    Pre-computed GBLUP prediction -> g_gblup
    |
    v
[Learnable Combination]
    alpha = sigmoid(learned_param)
    g_final = alpha * g_dl + (1 - alpha) * g_gblup
    |
    v
Output: Predicted GEBV
```

**PyTorch implementation:**

```python
class MoocleusDeepGBLUP(nn.Module):
    def __init__(self, n_snps: int, group_size: int = 50, hidden: int = 512):
        super().__init__()
        self.local_conv = nn.Conv1d(1, 4, kernel_size=group_size, stride=group_size)
        local_out = (n_snps // group_size) * 4

        self.fc = nn.Sequential(
            nn.Linear(local_out, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, snp_input, gblup_value):
        x = snp_input.unsqueeze(1)  # (batch, 1, n_snps)
        x = F.relu(self.local_conv(x))
        x = x.flatten(1)
        g_dl = self.fc(x)
        alpha = torch.sigmoid(self.alpha)
        return alpha * g_dl + (1 - alpha) * gblup_value.unsqueeze(1)
```

**Training:**
- 5-fold cross-validation, stratified by sire family to prevent data leakage
- MSE loss, Adam optimizer, lr=1e-3 with cosine annealing
- Early stopping on validation loss (patience=20)
- Report Pearson correlation (r) between predicted and true GEBVs

**Expected prediction accuracies (from published literature):**

| Trait | GBLUP r | deepGBLUP r | Improvement |
|---|---|---|---|
| Milk yield | 0.72 | 0.76 | +5.6% |
| Fat % | 0.68 | 0.73 | +7.4% |
| Protein % | 0.65 | 0.70 | +7.7% |
| Fertility | 0.35 | 0.38 | +8.6% |
| SCS (health) | 0.55 | 0.58 | +5.5% |
| Longevity | 0.40 | 0.44 | +10.0% |

### Composite Merit Index

After computing GEBVs for all 6 traits, combine them into a single "Moocleus Score" (analogous to USDA Net Merit $):

```python
def moocleus_score(gebvs: dict[str, float]) -> float:
    """Weighted composite index (weights approximate USDA NM$ relative emphasis)."""
    weights = {
        "milk_yield": 0.25,
        "fat_pct": 0.20,
        "protein_pct": 0.20,
        "fertility": 0.15,
        "scs": 0.10,
        "longevity": 0.10,
    }
    raw = sum(gebvs[trait] * w for trait, w in weights.items())
    # Normalize to 0-100 scale based on population distribution
    return normalize_to_percentile(raw)
```

---

## Demo Data Strategy

Since we cannot run real cattle genotyping, the demo uses simulation grounded in real biology.
This means Manhattan plots show real chromosomes, real gene names, and real genetic architecture.

### Step 1: Build Real SNP Panel

```python
# Cross-reference QTLdb associations with BovineHD chip positions
# Result: ~15,000 SNPs with real rsIDs, real chr:pos, real gene annotations
# Stored as moocleus_panel.json
```

### Step 2: Simulate Genotypes with LD Structure

```python
def simulate_herd_genotypes(n_animals=500, n_snps=15000, allele_freqs=None):
    """
    Uses real allele frequencies from BGVD.
    Applies linkage disequilibrium decay based on physical distance.
    Produces genotypes that look like real cattle data.
    """
    # For each SNP: sample from Binomial(2, p) with LD correlation to neighbors
    ...
```

### Step 3: Simulate Phenotypes from Real Genetic Architecture

```python
def simulate_trait(genotypes, causal_snps, effect_sizes, heritability):
    """
    causal_snps and effect_sizes come from QTLdb (REAL data).
    heritability from published estimates.
    phenotype = genetic_signal + environmental_noise
    """
    genetic_values = genotypes[:, causal_snps] @ effect_sizes
    var_env = np.var(genetic_values) * (1 - heritability) / heritability
    return genetic_values + np.random.normal(0, np.sqrt(var_env), len(genotypes))
```

### Step 4: Simulate Embryo Cohorts

```python
def simulate_embryos(sire_genotype, dam_genotype, n_embryos=12):
    """Mendelian segregation: each parent transmits one allele per locus."""
    # Result: 8-12 embryos per mating with realistic genetic variation
    ...
```

### Step 5: Pre-computed Demo Dataset

- 1 demo herd: 200 cows with phenotypes for all 6 traits
- 5 sire/dam matings producing 8-12 embryos each (~50 embryos total)
- Pre-computed GEBVs from all 3 models
- Notable SNPs flagged (DGAT1 K232A, ABCG2, GHR, CSN variants, SCD1)
- All stored as JSON + Parquet for fast Streamlit serving

### Key Real Genes to Highlight

| Gene | Chromosome | Effect | Why It Matters |
|---|---|---|---|
| DGAT1 (K232A) | BTA14 | +0.14% fat content | Most famous cattle QTL, large effect |
| ABCG2 (Y581S) | BTA6 | Milk yield + composition | Major pleiotropic effect |
| GHR | BTA20 | Milk yield | Growth hormone receptor |
| CSN1S1/CSN2/CSN3 | BTA6 | Protein content | Casein gene cluster |
| SCD1 | BTA26 | Fat composition | Fatty acid desaturation |
| PLAG1 | BTA14 | Stature + growth | Pleiotropic developmental gene |

---

## Streamlit App Structure

### Page 1: Home (`app/Home.py`)

- Hero section: "Moocleus Genomics" title + tagline
- Brief product description mirroring Nucleus: "Analyzes 15,000+ genetic markers across your embryo candidates to predict production, health, and longevity traits."
- "Explore Demo Herd" button (navigates to Dashboard)
- Three feature cards: "Polygenic Scores" / "Embryo Ranking" / "Model Comparison"
- Tech credibility footer: "Built on GBLUP + Deep Learning | Trained on real bovine GWAS data"

### Page 2: Dashboard (`app/pages/1_Dashboard.py`)

- Herd-level trait summary metrics (st.metric cards for each trait with deltas)
- Interactive Manhattan plot (Plotly scatter):
  - X-axis: chromosome position (BTA1-29 + X, alternating colors)
  - Y-axis: -log10(p-value) from GWAS
  - Genome-wide significance line at p < 5e-8
  - Hover tooltips: SNP ID, gene, p-value, effect size
  - Chromosome selector/filter
- Herd trait distribution histograms

### Page 3: Embryo Selection (`app/pages/2_Embryo_Selection.py`)

This is the money page — directly mirrors Nucleus Embryo.

- Grid of embryo cards showing:
  - Embryo ID with cow names (Daisy #4291, Buttercup #7183, Clover #3847)
  - Sire and dam identifiers
  - Overall Moocleus Score (0-100)
  - Mini radar chart (Plotly) with 6 trait axes
  - Color-coded badge: "Top Pick" (green) / "Above Average" (teal) / "Average" (gray)
- Sort by: Moocleus Score, individual trait GEBVs
- Comparison mode: select 2-4 embryos for side-by-side comparison
  - Overlaid radar chart
  - Trait-by-trait bar chart with delta values vs herd average

### Page 4: Embryo Report (`app/pages/3_Embryo_Report.py`)

- Select an embryo from dropdown
- Large circular Moocleus Score gauge
- Trait-by-trait breakdown:
  - GEBV value + percentile rank
  - Bell curve (normal distribution) showing where this embryo falls in the population
  - Breed comparison context
- Genetic highlights:
  - Notable SNP variants (e.g., "Carries favorable DGAT1 allele — associated with +0.14% fat content")
  - Carrier status for known recessive conditions (BLAD, CVM, Brachyspina)
- Inbreeding coefficient from GRM diagonal

### Page 5: Model Performance (`app/pages/4_Model_Performance.py`)

- Cross-validation accuracy table (r values for each model x trait)
- Bar chart comparing GBLUP vs Bayesian Ridge vs deepGBLUP
- Training details: dataset size, cross-validation strategy, hyperparameters
- "How It Works" explainer for each model tier

### Page 6: About (`app/pages/5_About.py`)

- Methodology explanation with citations
- Data sources with links
- "Why Cows?" section explaining the bovine-human genomics parallel
- The Nucleus connection: "Moocleus is to dairy cattle what Nucleus is to humans"
- References: VanRaden 2008, deepGBLUP (Lee et al. 2023), ReaGP, BANN

### Custom CSS Theme (`app/styles/theme.css`)

```css
/* Nucleus-inspired palette, adapted for bovine */
:root {
    --bg-primary: #FBF8F3;         /* Warm cream (from Nucleus) */
    --bg-card: #FFFFFF;
    --accent-primary: #2D6A4F;     /* Forest green (pastoral) */
    --accent-secondary: #719C9E;   /* Sage teal (from Nucleus) */
    --text-primary: #423C31;       /* Dark brown */
    --text-secondary: #8B8578;     /* Warm gray */
    --success: #40916C;
    --warning: #E9C46A;
    --danger: #E76F51;
}
```

---

## File Structure

```
moocleus-genomics/
├── CLAUDE.md                        # This file — project blueprint
├── README.md                        # Public-facing README
├── pyproject.toml                   # UV project config
├── .gitignore
├── Dockerfile
│
├── app/                             # Streamlit application
│   ├── Home.py                      # Landing / entry page
│   ├── pages/
│   │   ├── 1_Dashboard.py           # Herd overview + Manhattan plot
│   │   ├── 2_Embryo_Selection.py    # Embryo grid + comparison
│   │   ├── 3_Embryo_Report.py       # Individual embryo detail
│   │   ├── 4_Model_Performance.py   # Model accuracy + comparison
│   │   └── 5_About.py              # Methodology + citations
│   ├── components/
│   │   ├── charts.py                # Manhattan plot, radar, bell curve, gauge
│   │   ├── embryo_card.py           # Embryo display card component
│   │   └── metrics.py              # Trait summary metric cards
│   └── styles/
│       └── theme.css                # Custom CSS (Nucleus-inspired palette)
│
├── ml/                              # ML pipeline (separate from app)
│   ├── data/
│   │   ├── raw/                     # Downloaded datasets (gitignored)
│   │   │   ├── qtldb_cattle.txt
│   │   │   ├── bovinehd_manifest.csv
│   │   │   └── bgvd_allele_freqs.tsv
│   │   └── processed/
│   │       ├── moocleus_panel.json   # 15K curated SNP panel
│   │       ├── trait_weights/        # Per-trait SNP effect weights
│   │       │   ├── milk_yield.json
│   │       │   ├── fat_pct.json
│   │       │   ├── protein_pct.json
│   │       │   ├── fertility.json
│   │       │   ├── scs.json
│   │       │   └── longevity.json
│   │       └── simulated/
│   │           ├── herd_genotypes.npy
│   │           ├── herd_phenotypes.parquet
│   │           ├── embryo_genotypes.npy
│   │           └── embryo_gebvs.parquet
│   │
│   ├── pipeline/                    # Data processing scripts (run in order)
│   │   ├── 01_parse_qtl_data.py     # Parse Animal QTLdb download
│   │   ├── 02_build_snp_panel.py    # Cross-ref QTLdb + BovineHD -> Moocleus Panel
│   │   ├── 03_build_effect_weights.py  # Extract per-trait SNP weights
│   │   ├── 04_simulate_genotypes.py # Simulate 200-cow herd genotypes
│   │   ├── 05_simulate_phenotypes.py # Simulate trait phenotypes
│   │   ├── 06_simulate_embryos.py   # Mendelian segregation for 50 embryos
│   │   └── 07_export_demo_data.py   # Package data for Streamlit app
│   │
│   ├── models/                      # Model implementations
│   │   ├── gblup.py                 # VanRaden GRM + mixed model equations
│   │   ├── bayesian_ridge.py        # scikit-learn Bayesian Ridge wrapper
│   │   ├── deep_gblup.py            # PyTorch deepGBLUP (Conv1d + FC + GBLUP)
│   │   └── composite_index.py       # Multi-trait Moocleus Score computation
│   │
│   ├── training/                    # Training + evaluation scripts
│   │   ├── train_gblup.py           # Train GBLUP on simulated data
│   │   ├── train_deep_gblup.py      # Train deepGBLUP with CV
│   │   └── evaluate.py             # Cross-validation + accuracy reporting
│   │
│   ├── notebooks/                   # Jupyter notebooks (show the thought process)
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_gblup_walkthrough.ipynb
│   │   └── 03_model_comparison.ipynb
│   │
│   └── artifacts/                   # Trained model weights (gitignored except demo)
│       ├── gblup_solutions.pkl
│       ├── bayesian_ridge_weights.pkl
│       ├── deep_gblup_weights.pt
│       └── snp_effects_all_traits.json
│
└── scripts/
    ├── setup.sh                     # Install deps + download data
    └── run_pipeline.sh              # Run full ML pipeline end-to-end
```

---

## Build Phases

### Phase 1: Data Foundation

1. Download Animal QTLdb cattle data (tab-delimited)
2. Download BovineHD BeadChip manifest (CSV)
3. Download BGVD allele frequencies (TSV)
4. Run `01_parse_qtl_data.py` — parse and filter to Holstein-relevant associations
5. Run `02_build_snp_panel.py` — cross-reference QTLdb with BovineHD to create Moocleus Panel
6. Run `03_build_effect_weights.py` — extract per-trait SNP weights with LD clumping
7. Run `04_simulate_genotypes.py` — generate 200-cow herd with LD structure
8. Run `05_simulate_phenotypes.py` — simulate 6 traits using real genetic architecture
9. Run `06_simulate_embryos.py` — generate 50 embryos from 5 matings
10. Run `07_export_demo_data.py` — package everything for the app

### Phase 2: Core ML

1. Implement `gblup.py` — GRM construction + mixed model equations from scratch
2. Implement `bayesian_ridge.py` — wrapper around scikit-learn with SNP effect extraction
3. Train GBLUP and Bayesian Ridge on simulated data, verify accuracies
4. Implement `deep_gblup.py` — PyTorch model with locally-connected Conv1d + GBLUP combination
5. Train deepGBLUP with 5-fold CV, compare against GBLUP
6. Implement `composite_index.py` — Moocleus Score from multi-trait GEBVs
7. Create Jupyter notebooks documenting the entire process

### Phase 3: Streamlit App

1. Set up Streamlit project with multi-page layout
2. Create custom CSS theme (Nucleus-inspired palette)
3. Build Home page with hero and feature cards
4. Build Dashboard with trait metrics + interactive Manhattan plot
5. Build Embryo Selection page with cards, radar charts, comparison mode
6. Build Embryo Report page with score gauge, bell curves, SNP highlights
7. Build Model Performance page with accuracy tables and charts
8. Build About page with methodology and citations

### Phase 4: Polish

1. Add `.gitignore` for raw data, artifacts, venv
2. Write a proper README with demo screenshots
3. Dockerfile for containerized deployment
4. Deploy to Streamlit Cloud
5. Record a short demo GIF for README / X post

---

## Key References

- VanRaden, P.M. (2008). "Efficient methods to compute genomic predictions." *J. Dairy Sci.* 91:4414-4423.
- Lee, J. et al. (2023). "deepGBLUP: Joint deep learning and GBLUP framework for genomic prediction." *Genetics Selection Evolution* 55:25.
- Wiggans, G.R. et al. (2017). "Genomic selection in dairy cattle: the USDA experience." *Ann. Rev. Anim. Biosci.* 5:309-327.
- Hu, Z.L. et al. (2019). "Animal QTLdb: an improved database tool for livestock animal QTL/association data." *Nucleic Acids Res.* 47:D694-D698.
- Hayes, B.J. et al. (2019). "1000 Bull Genomes Project: towards genomic selection from whole-genome sequence data." *Anim. Genet.* 50:296-303.

---

## Commands

```bash
# Setup
uv sync                              # Install Python dependencies

# Data pipeline
python ml/pipeline/01_parse_qtl_data.py
python ml/pipeline/02_build_snp_panel.py
python ml/pipeline/03_build_effect_weights.py
python ml/pipeline/04_simulate_genotypes.py
python ml/pipeline/05_simulate_phenotypes.py
python ml/pipeline/06_simulate_embryos.py
python ml/pipeline/07_export_demo_data.py

# Training
python ml/training/train_gblup.py
python ml/training/train_deep_gblup.py
python ml/training/evaluate.py

# Run app
streamlit run app/Home.py

# Docker
docker build -t moocleus .
docker run -p 8501:8501 moocleus
```
