"""Moocleus Genomics — Individual Embryo Report."""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Embryo Report | Moocleus", page_icon="\U0001f9ec", layout="wide")

css_path = Path(__file__).parent.parent / "styles" / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from app.components.charts import create_bell_curve, create_radar_chart, create_score_gauge
from app.components.metrics import render_embryo_trait_table


@st.cache_data
def load_demo_data():
    data_path = Path(__file__).parent.parent.parent / "ml" / "data" / "processed" / "demo_data.json"
    with open(data_path) as f:
        return json.load(f)


data = load_demo_data()
trait_config = data["trait_config"]
embryos = data["embryos"]

# ── Embryo Selector ───────────────────────────────────────────
st.title("Embryo Report")

# Sort embryos by score (descending) for selector, but keep original order for ID mapping
embryos_sorted = sorted(embryos, key=lambda e: e["moocleus_score"], reverse=True)
embryo_options = {e['id']: e for e in embryos_sorted}

# Check if a specific embryo was selected from the grid
default_index = 0
if "selected_embryo_id" in st.session_state and st.session_state["selected_embryo_id"] in embryo_options:
    # Find the index of the selected embryo in the sorted list
    selected_id = st.session_state["selected_embryo_id"]
    default_index = list(embryo_options.keys()).index(selected_id)
    # Clear the session state so it doesn't persist on refresh
    del st.session_state["selected_embryo_id"]

selected_id = st.selectbox(
    "Select Embryo",
    list(embryo_options.keys()),
    index=default_index,  # Default to highest-scoring OR previously selected
    format_func=lambda eid: f"{eid} — {embryo_options[eid]['moocleus_score']:.0f} ({embryo_options[eid]['badge']})"
)
embryo = embryo_options[selected_id]

# ── Score + Identity ──────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    fig = create_score_gauge(embryo["moocleus_score"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    badge_class = {
        "Top Pick": "badge-top-pick",
        "Above Average": "badge-above-average",
        "Average": "badge-average",
    }.get(embryo["badge"], "badge-average")

    st.markdown(f"""
    ### {embryo['id']}
    **Parent 1:** {embryo['sire_id']} &nbsp;|&nbsp; **Parent 2:** {embryo['dam_id']}
    &nbsp;|&nbsp; **Mating Group:** {embryo['mating_group']}

    <span class="badge {badge_class}">{embryo['badge']}</span>
    """, unsafe_allow_html=True)

    # Radar chart
    trait_names = {k: v["name"] for k, v in trait_config.items()}
    fig = create_radar_chart(embryo["percentiles"], trait_names, height=280)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Trait-by-Trait Breakdown ──────────────────────────────────
st.subheader("Trait Breakdown")

render_embryo_trait_table(embryo, trait_config)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Population Bell Curves ────────────────────────────────────
st.subheader("Population Comparison")

trait_keys = list(trait_config.keys())
for row_start in range(0, len(trait_keys), 3):
    cols = st.columns(3)
    for i, col in enumerate(cols):
        idx = row_start + i
        if idx < len(trait_keys):
            tk = trait_keys[idx]
            ti = trait_config[tk]
            with col:
                # Use herd GEBV distribution (deepGBLUP) for bell curve
                herd_vals = data["herd"]["gebvs"]["deep_gblup"][tk]
                embryo_val = embryo["gebvs"][tk]
                fig = create_bell_curve(herd_vals, embryo_val, ti["name"], "GEBV")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Genetic Highlights ────────────────────────────────────────
st.subheader("Genetic Highlights")

if embryo["notable_snps"]:
    for snp in embryo["notable_snps"]:
        icon = {0: "\u2705", 1: "\u26A0\uFE0F", 2: "\u274C"}.get(snp["genotype"], "")
        st.markdown(
            f"**{snp['gene']}** ({snp['variant']}) &mdash; "
            f"{snp['genotype_desc']} {icon}\n\n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;_{snp['effect']}_"
        )
else:
    st.info("No notable SNP data available for this embryo.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Carrier Status ────────────────────────────────────────────
st.subheader("Recessive Condition Carrier Status")

carrier_cols = st.columns(4)
for i, (condition, is_carrier) in enumerate(embryo["carrier_status"].items()):
    with carrier_cols[i % 4]:
        if is_carrier:
            st.markdown(
                f"**{condition}**: <span class='carrier-positive'>Carrier</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"**{condition}**: <span class='carrier-clear'>Clear</span>",
                unsafe_allow_html=True,
            )

# ── Inbreeding ────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.metric("Inbreeding Coefficient (F)", f"{embryo['inbreeding_coeff']:.4f}")
if embryo["inbreeding_coeff"] > 0.0625:
    st.warning("Elevated inbreeding detected (F > 0.0625). Consider alternative mating.")

# ── Navigation ────────────────────────────────────────────────
st.markdown("---")
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
with nav_col1:
    if st.button("← Back to Candidates", use_container_width=True):
        st.switch_page("pages/2_Embryo_Selection.py")

with nav_col3:
    if st.button("Model Performance →", use_container_width=True):
        st.switch_page("pages/4_Model_Performance.py")
