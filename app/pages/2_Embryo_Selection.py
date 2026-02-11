"""Moocleus Genomics — Embryo Selection & Comparison."""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Embryo Selection | Moocleus", page_icon="\U0001f9ec", layout="wide")

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from app.components.charts import create_overlaid_radar, create_trait_comparison_bar
from app.components.embryo_card import render_embryo_card


@st.cache_data
def load_demo_data():
    data_path = Path(__file__).parent.parent.parent / "ml" / "data" / "processed" / "demo_data.json"
    with open(data_path) as f:
        return json.load(f)


data = load_demo_data()
trait_config = data["trait_config"]
embryos = data["embryos"]

# ── Header ────────────────────────────────────────────────────
st.title("Embryo Selection")
st.caption(f"{len(embryos)} embryos from {max(e['mating_group'] for e in embryos)} matings")

# ── Sort Controls ─────────────────────────────────────────────
sort_options = {"Moocleus Score": "moocleus_score"}
for tk, ti in trait_config.items():
    sort_options[ti["name"]] = tk

sort_by_name = st.selectbox("Sort by", list(sort_options.keys()))
sort_key = sort_options[sort_by_name]

if sort_key == "moocleus_score":
    embryos_sorted = sorted(embryos, key=lambda e: e["moocleus_score"], reverse=True)
else:
    embryos_sorted = sorted(
        embryos,
        key=lambda e: e["gebvs"].get(sort_key, 0),
        reverse=True,
    )

# ── Embryo Grid ───────────────────────────────────────────────
n_cols = 4
for row_start in range(0, len(embryos_sorted), n_cols):
    cols = st.columns(n_cols)
    for i, col in enumerate(cols):
        idx = row_start + i
        if idx < len(embryos_sorted):
            with col:
                render_embryo_card(embryos_sorted[idx], trait_config)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Comparison Mode ───────────────────────────────────────────
st.subheader("Compare Embryos")
embryo_names = [e["name"] for e in embryos]
selected_names = st.multiselect(
    "Select 2-4 embryos to compare",
    embryo_names,
    max_selections=4,
)

if len(selected_names) >= 2:
    selected_embryos = [e for e in embryos if e["name"] in selected_names]
    trait_names = {k: v["name"] for k, v in trait_config.items()}

    # Overlaid radar chart
    st.markdown("**Trait Profile Comparison**")
    fig = create_overlaid_radar(selected_embryos, trait_names)
    st.plotly_chart(fig, use_container_width=True)

    # Trait-by-trait bar charts
    st.markdown("**Trait Deltas vs. Herd Average**")
    trait_keys = list(trait_config.keys())
    for row_start in range(0, len(trait_keys), 3):
        cols = st.columns(3)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(trait_keys):
                tk = trait_keys[idx]
                ti = trait_config[tk]
                herd_mean = data["herd"]["stats"][tk]["mean"]
                with col:
                    fig = create_trait_comparison_bar(
                        selected_embryos, tk, ti["name"], herd_mean,
                    )
                    st.plotly_chart(fig, use_container_width=True)
elif len(selected_names) == 1:
    st.info("Select at least 2 embryos to compare.")
