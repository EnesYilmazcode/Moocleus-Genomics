"""Metric card components for the Moocleus Genomics app."""

import streamlit as st


def render_trait_metrics(herd_data: dict, trait_config: dict):
    """
    Render a row of st.metric cards for each trait.

    Args:
        herd_data: dict with "phenotypes" -> {trait_key: [values]}
        trait_config: dict mapping trait_key -> {name, unit, h2, ...}
    """
    cols = st.columns(len(trait_config))
    for col, (trait_key, trait_info) in zip(cols, trait_config.items()):
        values = herd_data["phenotypes"].get(trait_key, [])
        if values:
            mean_val = sum(values) / len(values)
        else:
            mean_val = 0

        with col:
            st.metric(
                label=trait_info["name"],
                value=f"{mean_val:.1f}",
                delta=f"h\u00B2 = {trait_info['h2']:.2f}",
            )


def render_embryo_trait_table(embryo: dict, trait_config: dict):
    """
    Render a trait-by-trait breakdown for a single embryo.

    Args:
        embryo: dict with gebvs, percentiles
        trait_config: dict mapping trait_key -> {name, unit, ...}
    """
    for trait_key, trait_info in trait_config.items():
        gebv = embryo["gebvs"].get(trait_key, 0)
        pct = embryo["percentiles"].get(trait_key, 50)

        # Color-code by percentile
        if pct >= 75:
            color = "#40916C"
        elif pct >= 50:
            color = "#719C9E"
        elif pct >= 25:
            color = "#E9C46A"
        else:
            color = "#E76F51"

        cols = st.columns([3, 2, 2])
        with cols[0]:
            st.markdown(f"**{trait_info['name']}**")
        with cols[1]:
            st.markdown(f"GEBV: `{gebv:.3f}`")
        with cols[2]:
            st.markdown(
                f"<span style='color: {color}; font-weight: 600;'>P{pct}</span>",
                unsafe_allow_html=True,
            )
