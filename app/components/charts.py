"""
Shared Plotly chart factory functions for the Moocleus Genomics app.
"""

import numpy as np
import plotly.graph_objects as go

# Moocleus color palette
COLORS = {
    "primary": "#2D6A4F",
    "secondary": "#719C9E",
    "success": "#40916C",
    "warning": "#E9C46A",
    "danger": "#E76F51",
    "text": "#423C31",
    "text_light": "#8B8578",
    "bg": "#FBF8F3",
}

# Alternating chromosome colors for Manhattan plot
CHR_COLORS = ["#2D6A4F", "#719C9E"]

TRAIT_COLORS = {
    "milk_yield": "#2D6A4F",
    "fat_pct": "#E9C46A",
    "protein_pct": "#719C9E",
    "fertility": "#E76F51",
    "scs": "#8B8578",
    "longevity": "#40916C",
}

MODEL_COLORS = {
    "gblup": "#719C9E",
    "bayesian_ridge": "#E9C46A",
    "deep_gblup": "#2D6A4F",
}


def _transparent_layout() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="sans-serif"),
        margin=dict(l=50, r=20, t=40, b=40),
    )


def create_manhattan_plot(
    manhattan_data: dict,
    trait: str,
    selected_chr: str | None = None,
) -> go.Figure:
    """Interactive Manhattan plot with alternating chromosome colors."""
    data = manhattan_data.get(trait, {})
    if not data or not data.get("chr"):
        fig = go.Figure()
        fig.update_layout(title="No data available", **_transparent_layout())
        return fig

    chroms = data["chr"]
    positions = data["pos"]
    neg_log_p = data["neg_log_p"]
    snp_ids = data["snp_id"]
    genes = data["gene"]
    effects = data["effect"]

    # Build chromosome order for x-axis
    chr_order = list(range(1, 30)) + ["X"]
    chr_order_map = {c: i for i, c in enumerate(chr_order)}

    # Filter by chromosome if selected
    if selected_chr and selected_chr != "All":
        try:
            chr_val = int(selected_chr)
        except ValueError:
            chr_val = selected_chr
        mask = [c == chr_val for c in chroms]
        chroms = [c for c, m in zip(chroms, mask) if m]
        positions = [p for p, m in zip(positions, mask) if m]
        neg_log_p = [v for v, m in zip(neg_log_p, mask) if m]
        snp_ids = [s for s, m in zip(snp_ids, mask) if m]
        genes = [g for g, m in zip(genes, mask) if m]
        effects = [e for e, m in zip(effects, mask) if m]

    # Compute cumulative positions for genome-wide x-axis
    if not selected_chr or selected_chr == "All":
        chr_offsets = {}
        cumulative = 0
        chr_lengths_approx = {}
        for c in chr_order:
            chr_positions = [p for ch, p in zip(data["chr"], data["pos"]) if ch == c]
            chr_lengths_approx[c] = max(chr_positions) if chr_positions else 0
            chr_offsets[c] = cumulative
            cumulative += chr_lengths_approx[c]

        x_vals = [chr_offsets.get(c, 0) + p for c, p in zip(chroms, positions)]
    else:
        x_vals = positions

    # Color by chromosome
    colors = [CHR_COLORS[chr_order_map.get(c, 0) % 2] for c in chroms]

    # Highlight significant SNPs
    sig_threshold = 7.3  # -log10(5e-8)
    sizes = [8 if v > sig_threshold else 4 for v in neg_log_p]

    hover_text = []
    for i in range(len(chroms)):
        gene_str = f"<br>Gene: {genes[i]}" if genes[i] else ""
        hover_text.append(
            f"SNP: {snp_ids[i]}<br>"
            f"Chr: {chroms[i]}<br>"
            f"Pos: {positions[i]:,}<br>"
            f"-log10(p): {neg_log_p[i]:.2f}<br>"
            f"Effect: {effects[i]:.4f}"
            f"{gene_str}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=neg_log_p,
        mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.7),
        text=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))

    # Genome-wide significance line
    fig.add_hline(
        y=sig_threshold,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text="p = 5e-8",
        annotation_position="top right",
    )

    # Suggestive significance line
    fig.add_hline(
        y=5.0,
        line_dash="dot",
        line_color=COLORS["text_light"],
        opacity=0.5,
    )

    layout = _transparent_layout()
    layout.update(
        title=None,
        xaxis_title="Chromosome" if not selected_chr or selected_chr == "All" else f"Position (BTA{selected_chr})",
        yaxis_title="-log10(p-value)",
        height=400,
        hovermode="closest",
    )

    # Add chromosome labels if showing all
    if not selected_chr or selected_chr == "All":
        layout["xaxis"] = dict(
            showticklabels=True,
            tickmode="array",
            tickvals=[chr_offsets.get(c, 0) + chr_lengths_approx.get(c, 0) / 2 for c in chr_order if chr_lengths_approx.get(c, 0) > 0],
            ticktext=[str(c) for c in chr_order if chr_lengths_approx.get(c, 0) > 0],
            title="Chromosome",
        )

    fig.update_layout(**layout)
    return fig


def create_radar_chart(
    percentiles: dict[str, int],
    trait_names: dict[str, str],
    height: int = 300,
) -> go.Figure:
    """6-axis radar chart for embryo trait profile using percentile values."""
    categories = list(trait_names.values())
    values = [percentiles.get(k, 50) for k in trait_names]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor=f"rgba(45, 106, 79, 0.15)",
        line=dict(color=COLORS["primary"], width=2),
        name="Percentile",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=height,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color=COLORS["text"]),
    )
    return fig


def create_bell_curve(
    population_values: list[float],
    embryo_value: float,
    trait_name: str,
    unit: str,
) -> go.Figure:
    """Normal distribution showing where an embryo falls in the population."""
    pop = np.array(population_values)
    mu, sigma = pop.mean(), pop.std()
    if sigma == 0:
        sigma = 1

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    percentile = int(round(np.mean(pop <= embryo_value) * 100))

    fig = go.Figure()

    # Population distribution
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=COLORS["secondary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(113, 156, 158, 0.15)",
        name="Population",
        showlegend=False,
    ))

    # Embryo position
    embryo_y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((embryo_value - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(
        x=[embryo_value, embryo_value],
        y=[0, embryo_y],
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name="This Embryo",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=[embryo_value],
        y=[embryo_y],
        mode="markers+text",
        marker=dict(color=COLORS["primary"], size=10),
        text=[f"P{percentile}"],
        textposition="top center",
        textfont=dict(color=COLORS["primary"], size=12, family="sans-serif"),
        showlegend=False,
    ))

    layout = _transparent_layout()
    layout.update(
        xaxis_title=f"{trait_name} ({unit})" if unit else trait_name,
        yaxis=dict(visible=False),
        height=200,
        margin=dict(l=20, r=20, t=10, b=40),
    )
    fig.update_layout(**layout)
    return fig


def create_score_gauge(score: float) -> go.Figure:
    """Circular gauge for Moocleus Score (0-100)."""
    # Color based on score
    if score >= 75:
        color = COLORS["success"]
    elif score >= 50:
        color = COLORS["secondary"]
    elif score >= 25:
        color = COLORS["warning"]
    else:
        color = COLORS["danger"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=48, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=COLORS["text_light"]),
            bar=dict(color=color),
            bgcolor="white",
            borderwidth=2,
            bordercolor=COLORS["text_light"],
            steps=[
                dict(range=[0, 25], color="#f5f0e8"),
                dict(range=[25, 50], color="#efe9dd"),
                dict(range=[50, 75], color="#e8e2d5"),
                dict(range=[75, 100], color="#e0daca"),
            ],
        ),
        title=dict(text="Moocleus Score", font=dict(size=16, color=COLORS["text"])),
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_trait_comparison_bar(
    embryos: list[dict],
    trait_key: str,
    trait_name: str,
    herd_mean: float,
) -> go.Figure:
    """Grouped bar chart comparing embryos on a single trait with delta vs herd average."""
    names = [e["name"] for e in embryos]
    values = [e["gebvs"][trait_key] for e in embryos]
    deltas = [v - herd_mean for v in values]

    colors = [COLORS["success"] if d >= 0 else COLORS["danger"] for d in deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=deltas,
        marker_color=colors,
        text=[f"{d:+.2f}" for d in deltas],
        textposition="outside",
        showlegend=False,
    ))

    fig.add_hline(y=0, line_color=COLORS["text_light"], line_width=1)

    layout = _transparent_layout()
    layout.update(
        title=f"{trait_name} (vs. Herd Average)",
        yaxis_title="Delta GEBV",
        height=300,
    )
    fig.update_layout(**layout)
    return fig


def create_model_comparison_chart(
    model_performance: dict,
    trait_config: dict,
) -> go.Figure:
    """Grouped bar chart: models x traits showing Pearson r accuracy."""
    trait_keys = list(trait_config.keys())
    trait_labels = [trait_config[t]["name"] for t in trait_keys]

    fig = go.Figure()
    model_labels = {"gblup": "GBLUP", "bayesian_ridge": "Bayesian Ridge", "deep_gblup": "deepGBLUP"}

    for model_key, model_name in model_labels.items():
        values = [model_performance.get(model_key, {}).get(t, 0) for t in trait_keys]
        fig.add_trace(go.Bar(
            name=model_name,
            x=trait_labels,
            y=values,
            marker_color=MODEL_COLORS.get(model_key, COLORS["primary"]),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    layout = _transparent_layout()
    layout.update(
        barmode="group",
        yaxis_title="Prediction Accuracy (r)",
        yaxis_range=[0, 1.0],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_layout(**layout)
    return fig


def create_trait_distribution_histogram(
    values: list[float],
    trait_name: str,
    unit: str,
) -> go.Figure:
    """Histogram of herd trait values."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=25,
        marker_color=COLORS["secondary"],
        opacity=0.75,
        showlegend=False,
    ))

    layout = _transparent_layout()
    layout.update(
        title=trait_name,
        xaxis_title=unit if unit else trait_name,
        yaxis_title="Count",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_layout(**layout)
    return fig


def create_overlaid_radar(
    embryos: list[dict],
    trait_names: dict[str, str],
) -> go.Figure:
    """Overlaid radar chart comparing multiple embryos."""
    categories = list(trait_names.values())
    categories_closed = categories + [categories[0]]

    colors = [COLORS["primary"], COLORS["danger"], COLORS["warning"], COLORS["secondary"]]

    fig = go.Figure()
    for i, embryo in enumerate(embryos):
        values = [embryo["percentiles"].get(k, 50) for k in trait_names]
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill="toself",
            fillcolor=f"rgba({','.join(str(int(colors[i % len(colors)].lstrip('#')[j:j+2], 16)) for j in (0, 2, 4))}, 0.1)",
            line=dict(color=colors[i % len(colors)], width=2),
            name=embryo["name"],
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig
