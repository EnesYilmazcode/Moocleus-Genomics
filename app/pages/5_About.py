"""Moocleus Genomics — About & Methodology."""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="About | Moocleus", page_icon="\U0001f9ec", layout="wide")

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.title("About Moocleus Genomics")

# ── Why Cows? ─────────────────────────────────────────────────
st.subheader("Why Cows?")
st.markdown("""
Bovine genomics is actually *ahead* of human genomics in genomic prediction.
Cattle have larger genotyped training populations (6.6M+ Holsteins), higher
prediction accuracies for production traits, and no ethical constraints on
selection. This makes dairy cattle the ideal proving ground for polygenic
prediction methods.

Where human polygenic risk scores explain 1-5% of variance for most diseases,
bovine genomic predictions achieve **60-80% accuracy** for production traits
like milk yield and fat percentage. The same statistical and deep learning
methods work in both species \u2014 cattle just have better training data.
""")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── The Nucleus Connection ────────────────────────────────────
st.subheader("The Nucleus Connection")
st.markdown("""
Moocleus is a bovine analog of [Nucleus Genomics](https://mynucleus.com/) \u2014 a
direct-to-consumer genomics company that calculates polygenic risk scores for
humans and offers embryo screening via their "Origin" models.
""")

st.markdown("""
| | **Nucleus Genomics** (Humans) | **Moocleus Genomics** (Cattle) |
|---|---|---|
| **Scores** | Polygenic Risk Scores for 800+ diseases | GEBVs for 6 dairy traits |
| **Screening** | Origin embryo screening models | Embryo selection + ranking engine |
| **Markers** | 7M genetic markers analyzed | 15K informative markers (Moocleus Panel) |
| **Methods** | Deep learning + statistical hybrids | deepGBLUP + GBLUP hybrid |
| **Interface** | Consumer web dashboard | Streamlit app with Plotly visualizations |
""")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Data Sources ──────────────────────────────────────────────
st.subheader("Data Sources")
st.markdown("""
All datasets used are publicly available:

- **[Animal QTLdb](https://www.animalgenome.org/cgi-bin/QTLdb/index)** \u2014
  5,920+ cattle QTL/association data points with trait-SNP mappings. Provides
  the genetic architecture for realistic phenotype simulation.

- **[Illumina BovineHD BeadChip](https://www.illumina.com/products/by-type/microarray-kits/bovinehd.html)** \u2014
  777,962 SNP positions across all 29 autosomes + X. Defines the "chip" used to
  build the Moocleus Panel.

- **[Bovine Genome Variation Database (BGVD)](http://animal.omics.pro/code/index.php/BosVar)** \u2014
  ~60.44M SNPs with minor allele frequencies across 54 cattle breeds.

- **[Bovine Genome Database (BGD)](https://bovinegenome.elsiklab.missouri.edu/)** \u2014
  ARS-UCD2.0 assembly with gene annotations, QTL data, and RNA-seq tracks.

- **[1000 Bull Genomes Project](https://db.cngb.org/)** \u2014
  2,703 whole-genome sequences from diverse cattle breeds, 84M SNPs.
""")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Methodology ───────────────────────────────────────────────
st.subheader("Methodology")
st.markdown("""
**Trait Prediction Pipeline:**
1. **SNP Panel Construction** \u2014 15,000 informative markers curated from QTLdb
   associations cross-referenced with BovineHD chip positions
2. **Genotype Simulation** \u2014 200-cow reference herd with linkage disequilibrium
   structure based on real allele frequencies
3. **Phenotype Simulation** \u2014 6 dairy traits generated from causal SNP architecture
   with published heritabilities
4. **Model Training** \u2014 GBLUP, Bayesian Ridge, and deepGBLUP trained with 5-fold
   cross-validation
5. **Embryo Screening** \u2014 Mendelian segregation from selected sire-dam pairs,
   GEBVs predicted by all three models

**Traits Analyzed:**

| Trait | Heritability | Description |
|---|---|---|
| Milk Yield | 0.28 | 305-day mature-equivalent milk yield (kg/lactation) |
| Fat % | 0.26 | Milk fat percentage |
| Protein % | 0.23 | Milk protein percentage |
| Fertility | 0.04 | Daughter pregnancy rate (standardized) |
| Somatic Cell Score | 0.12 | Indicator of udder health (lower is better) |
| Longevity | 0.05 | Productive life (standardized) |

**Composite Index:** The Moocleus Score combines all 6 trait GEBVs using weights
approximating USDA Net Merit $ relative emphasis (production: 65%, health: 10%,
fertility: 15%, longevity: 10%).
""")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── References ────────────────────────────────────────────────
st.subheader("References")
st.markdown("""
1. VanRaden, P.M. (2008). "Efficient methods to compute genomic predictions."
   *J. Dairy Sci.* 91:4414-4423.

2. Lee, J. et al. (2023). "deepGBLUP: Joint deep learning and GBLUP framework
   for genomic prediction." *Genetics Selection Evolution* 55:25.

3. Wiggans, G.R. et al. (2017). "Genomic selection in dairy cattle: the USDA
   experience." *Ann. Rev. Anim. Biosci.* 5:309-327.

4. Hu, Z.L. et al. (2019). "Animal QTLdb: an improved database tool for livestock
   animal QTL/association data." *Nucleic Acids Res.* 47:D694-D698.

5. Hayes, B.J. et al. (2019). "1000 Bull Genomes Project: towards genomic selection
   from whole-genome sequence data." *Anim. Genet.* 50:296-303.
""")

st.markdown("""
<div class="tech-footer">
    Moocleus Genomics &mdash; Breed Better. Know More.
</div>
""", unsafe_allow_html=True)
