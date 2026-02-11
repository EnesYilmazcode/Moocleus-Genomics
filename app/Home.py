"""Moocleus Genomics — Home Page."""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Moocleus Genomics",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load custom CSS
css_path = Path(__file__).parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1>Moocleus Genomics</h1>
    <p class="hero-tagline">Breed Better. Know More.</p>
    <p class="hero-description">
        Analyzes 15,000+ genetic markers across your embryo candidates to predict
        production, health, and longevity traits using GBLUP and deep learning hybrids.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Feature Cards ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">\U0001f4ca</div>
        <h3>Polygenic Scores</h3>
        <p>Genomic Estimated Breeding Values for 6 key dairy traits computed from
        15,000 informative SNP markers across all 29 bovine autosomes.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">\U0001f3af</div>
        <h3>Embryo Ranking</h3>
        <p>Compare and rank embryo candidates with our composite Moocleus Score,
        radar charts, and trait-by-trait breakdowns with population percentiles.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">\U0001f9e0</div>
        <h3>Model Comparison</h3>
        <p>Three prediction tiers: GBLUP (industry standard), Bayesian Ridge
        (SNP-level effects), and deepGBLUP (deep learning + GBLUP hybrid).</p>
    </div>
    """, unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    if st.button("Explore Demo Herd", type="primary", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="tech-footer">
    Built on GBLUP + Deep Learning &nbsp;|&nbsp; Trained on real bovine GWAS data &nbsp;|&nbsp;
    <a href="https://www.animalgenome.org/cgi-bin/QTLdb/index" target="_blank">Animal QTLdb</a> &nbsp;|&nbsp;
    VanRaden (2008) &bull; Lee et al. (2023)
</div>
""", unsafe_allow_html=True)
