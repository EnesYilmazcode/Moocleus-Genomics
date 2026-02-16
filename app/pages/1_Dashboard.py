"""Moocleus Genomics — Herd Dashboard with Manhattan Plot."""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Dashboard | Moocleus", page_icon="\U0001f9ec", layout="wide")

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from app.components.charts import (
    create_manhattan_plot,
    create_trait_distribution_histogram,
)
from app.components.metrics import render_trait_metrics


@st.cache_data
def load_demo_data():
    data_path = Path(__file__).parent.parent.parent / "ml" / "data" / "processed" / "demo_data.json"
    with open(data_path) as f:
        return json.load(f)


data = load_demo_data()
trait_config = data["trait_config"]

# ── Header ────────────────────────────────────────────────────
st.title("Herd Dashboard")
st.caption(f"{data['herd']['n_animals']} animals | {data['panel']['n_snps']:,} SNP markers")

# ── Trait Summary Metrics ─────────────────────────────────────
render_trait_metrics(data["herd"], trait_config)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Manhattan Plot ────────────────────────────────────────────
st.subheader("Genome-Wide Association")

col1, col2 = st.columns([2, 1])
with col1:
    trait_options = {v["name"]: k for k, v in trait_config.items()}
    selected_trait_name = st.selectbox("Select trait", list(trait_options.keys()))
    selected_trait = trait_options[selected_trait_name]
with col2:
    chr_options = ["All"] + [str(c) for c in range(1, 30)] + ["X"]
    chr_filter = st.selectbox("Chromosome", chr_options)

fig = create_manhattan_plot(
    data["manhattan"],
    selected_trait,
    chr_filter if chr_filter != "All" else None,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Trait Distributions ───────────────────────────────────────
st.subheader("Herd Trait Distributions")

trait_keys = list(trait_config.keys())
for row_start in range(0, len(trait_keys), 3):
    cols = st.columns(3)
    for i, col in enumerate(cols):
        idx = row_start + i
        if idx < len(trait_keys):
            tk = trait_keys[idx]
            ti = trait_config[tk]
            with col:
                fig = create_trait_distribution_histogram(
                    data["herd"]["phenotypes"][tk],
                    ti["name"],
                    ti["unit"],
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Navigation ───────────────────────────────────────────────
st.markdown("---")
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    if st.button("View Embryo Candidates \u2192", type="primary", use_container_width=True):
        st.switch_page("pages/2_Embryo_Selection.py")
