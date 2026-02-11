"""Moocleus Genomics — Model Performance Comparison."""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Model Performance | Moocleus", page_icon="\U0001f9ec", layout="wide")

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from app.components.charts import create_model_comparison_chart


@st.cache_data
def load_demo_data():
    data_path = Path(__file__).parent.parent.parent / "ml" / "data" / "processed" / "demo_data.json"
    with open(data_path) as f:
        return json.load(f)


data = load_demo_data()
trait_config = data["trait_config"]
model_perf = data["model_performance"]

# ── Header ────────────────────────────────────────────────────
st.title("Model Performance")
st.caption("Prediction accuracy measured as Pearson correlation (r) between predicted and true genetic values")

# ── Accuracy Table ────────────────────────────────────────────
st.subheader("Prediction Accuracy by Trait")

rows = []
for tk, ti in trait_config.items():
    row = {"Trait": ti["name"], "h\u00B2": ti["h2"]}
    for model_key, model_name in [("gblup", "GBLUP"), ("bayesian_ridge", "Bayesian Ridge"), ("deep_gblup", "deepGBLUP")]:
        row[model_name] = model_perf.get(model_key, {}).get(tk, 0)
    rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(
    df.style.format({
        "h\u00B2": "{:.2f}",
        "GBLUP": "{:.4f}",
        "Bayesian Ridge": "{:.4f}",
        "deepGBLUP": "{:.4f}",
    }).highlight_max(axis=1, subset=["GBLUP", "Bayesian Ridge", "deepGBLUP"], color="#d4edda"),
    use_container_width=True,
    hide_index=True,
)

# ── Bar Chart ─────────────────────────────────────────────────
st.subheader("Model Comparison")
fig = create_model_comparison_chart(model_perf, trait_config)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Training Details ──────────────────────────────────────────
st.subheader("Training Details")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Animals", "200")
with col2:
    st.metric("SNP Markers", "15,000")
with col3:
    st.metric("Cross-Validation", "5-fold")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────
st.subheader("How It Works")

with st.expander("Tier 1: GBLUP \u2014 Genomic Best Linear Unbiased Prediction"):
    st.markdown("""
    The industry gold standard used by every national cattle breeding program.

    **Model:** `y = Xb + Za + e` where `a ~ N(0, G * sigma_a^2)`

    **Key steps:**
    1. Construct the Genomic Relationship Matrix (GRM) using VanRaden (2008) Method 1
    2. Solve Henderson's Mixed Model Equations
    3. Extract Genomic Estimated Breeding Values (GEBVs)

    **Strengths:** Theoretically optimal for infinitesimal model; captures all genetic signal
    through the GRM without needing to identify individual SNP effects.

    **Reference:** VanRaden, P.M. (2008). *J. Dairy Sci.* 91:4414-4423.
    """)

with st.expander("Tier 2: Bayesian Ridge Regression"):
    st.markdown("""
    Estimates individual SNP effects, enabling per-marker analysis and Manhattan plots.

    **Model:** `phenotype = genotype_matrix @ snp_effects + intercept`

    **Key features:**
    - L2-regularized regression with automatic relevance determination
    - Produces effect size estimates for every SNP
    - Used to generate Manhattan plot data (-log10 p-values from effect z-scores)

    **Strengths:** Direct SNP effect estimation; computationally efficient; provides
    interpretable weights for polygenic score calculation.
    """)

with st.expander("Tier 3: deepGBLUP \u2014 Deep Learning Hybrid"):
    st.markdown("""
    Combines deep learning's ability to capture non-linear patterns with GBLUP's
    statistical optimality.

    **Architecture:**
    - **Local feature extraction:** Conv1d(1, 4, kernel=50, stride=50) learns haplotype patterns
    from groups of 50 adjacent SNPs
    - **Deep pathway:** FC(1200 -> 512 -> 128 -> 1) with BatchNorm + Dropout
    - **GBLUP branch:** Pre-computed GBLUP GEBV
    - **Combination:** `output = alpha * g_deep + (1 - alpha) * g_gblup` where alpha is learned

    **Training:** 5-fold CV, MSE loss, Adam optimizer (lr=1e-3), cosine annealing, early stopping (patience=20)

    **Reference:** Lee, J. et al. (2023). *Genetics Selection Evolution* 55:25.
    """)
