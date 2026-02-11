"""Embryo display card component for the Streamlit app."""

import streamlit as st

from app.components.charts import create_radar_chart


def render_embryo_card(embryo: dict, trait_config: dict):
    """
    Render a single embryo as a styled card with mini radar chart.

    Args:
        embryo: dict with id, name, sire_id, dam_id, moocleus_score, badge,
                gebvs, percentiles
        trait_config: dict mapping trait_key -> {"name": str, ...}
    """
    badge_class = {
        "Top Pick": "badge-top-pick",
        "Above Average": "badge-above-average",
        "Average": "badge-average",
    }.get(embryo["badge"], "badge-average")

    st.markdown(f"""
    <div class="embryo-card">
        <div class="embryo-header">
            <span class="embryo-name">{embryo['name']}</span>
            <span class="badge {badge_class}">{embryo['badge']}</span>
        </div>
        <div class="embryo-score">{embryo['moocleus_score']:.0f}</div>
        <div class="embryo-parents">{embryo['sire_id']} &times; {embryo['dam_id']}</div>
    </div>
    """, unsafe_allow_html=True)

    trait_names = {k: v["name"] for k, v in trait_config.items()}
    fig = create_radar_chart(embryo["percentiles"], trait_names, height=220)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
