"""Moocleus Genomics — Pipeline Overview (Landing Page)."""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Moocleus Genomics",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="collapsed",
)

css_path = Path(__file__).parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_demo_data():
    data_path = Path(__file__).parent.parent / "ml" / "data" / "processed" / "demo_data.json"
    with open(data_path) as f:
        return json.load(f)


data = load_demo_data()
panel = data["panel"]
herd = data["herd"]
embryos = data["embryos"]
model_perf = data["model_performance"]
trait_config = data["trait_config"]

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1>Moocleus Genomics</h1>
    <p class="hero-tagline">Genomic embryo selection for dairy cattle</p>
</div>
""", unsafe_allow_html=True)

# ── Data Source ──────────────────────────────────────────────
st.markdown("### Data Source")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sequenced Bulls", f"{panel.get('real_samples', 1842):,}")
col2.metric("Mitochondrial Variants", f"{panel.get('real_variants', 4540):,}")
col3.metric("SNP Panel", f"{panel['n_snps']:,} markers")
col4.metric("Dairy Traits", f"{len(trait_config)}")

st.info(
    f"**{panel.get('source', '1000 Bull Genomes Project (PRJEB42783)')}** "
    f"| Allele frequencies derived from real whole-genome sequencing of {panel.get('real_samples', 1842)} "
    f"bulls across Bos taurus, Bos indicus, and crossbreeds "
    f"| Phenotypes simulated using published heritabilities and known QTL effects"
)

# ── Pipeline Steps ───────────────────────────────────────────
st.markdown("### Pipeline")

steps = st.columns(4)
with steps[0]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">1</div>
        <h3>Extract</h3>
        <p>Parse VCF genotypes from the 1000 Bull Genomes Project via ENA API.
        Supports cyvcf2 (Linux) and scikit-allel (Windows).</p>
    </div>
    """, unsafe_allow_html=True)

with steps[1]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">2</div>
        <h3>Predict</h3>
        <p>Compute Genomic Estimated Breeding Values (GEBVs) across 6 dairy traits
        using GBLUP with a genomic relationship matrix.</p>
    </div>
    """, unsafe_allow_html=True)

with steps[2]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">3</div>
        <h3>Rank</h3>
        <p>Score and rank embryo candidates using a weighted composite index
        balancing production, health, and longevity traits.</p>
    </div>
    """, unsafe_allow_html=True)

with steps[3]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">4</div>
        <h3>Select</h3>
        <p>Compare top candidates side-by-side with radar charts, trait deltas,
        carrier status, and population percentiles.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Quick Stats ──────────────────────────────────────────────
st.markdown("### Results Overview")
col1, col2, col3 = st.columns(3)

top_picks = [e for e in embryos if e["badge"] == "Top Pick"]
above_avg = [e for e in embryos if e["badge"] == "Above Average"]
best = max(embryos, key=lambda e: e["moocleus_score"])

col1.metric("Embryo Candidates", len(embryos))
col2.metric("Top Picks", len(top_picks))
col3.metric("Best Score", f"{best['moocleus_score']:.0f} ({best['id']})")

# Model accuracy summary
st.markdown("**Model Accuracies (correlation with phenotype)**")
perf_cols = st.columns(3)
for i, (model, label) in enumerate([
    ("gblup", "GBLUP"), ("bayesian_ridge", "Bayesian Ridge"), ("deep_gblup", "deepGBLUP")
]):
    with perf_cols[i]:
        milk_r = model_perf[model]["milk_yield"]
        fat_r = model_perf[model]["fat_pct"]
        st.markdown(f"**{label}**: milk r={milk_r:.2f}, fat r={fat_r:.2f}")

# ── Navigation ───────────────────────────────────────────────
st.markdown("---")
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    if st.button("View Herd Analysis \u2192", type="primary", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")
