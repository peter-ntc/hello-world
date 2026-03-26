import plotly.graph_objects as go
from macro_data import yields_df, spread_df, pe_df


def apply_dark_chart_layout(
    fig,
    height: int = 210,
    show_legend: bool = True,
    top_margin: int = 10,
    bottom_margin: int = 34,
    left_margin: int = 44,
    right_margin: int = 12,
):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
        font=dict(
            family="Inter, Arial, sans-serif",
            size=10,
            color="#d9e3f2",
        ),
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            font=dict(size=8, color="#d9e3f2"),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
            traceorder="normal",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="rgba(185,205,236,0.65)",
        linewidth=1,
        tickfont=dict(size=8, color="#d9e3f2"),
        tickcolor="rgba(0,0,0,0)",
        title_font=dict(size=8, color="#d9e3f2"),
        ticks="outside",
        ticklen=4,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(157,190,238,0.45)",
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont=dict(size=8, color="#d9e3f2"),
        tickcolor="rgba(0,0,0,0)",
        title_font=dict(size=8, color="#d9e3f2"),
    )

    return fig


def build_yield_chart(height: int = 210):
    fig = go.Figure()

    # Match PDF legend order and colors
    series_order = [
        "US High Yield",
        "US BB",
        "US BBB",
        "Eur High Yield",
    ]

    series_colors = {
        "US High Yield": "#14284b",   # dark navy
        "US BB": "#7ea4d8",           # light blue
        "US BBB": "#1d4fa1",          # mid blue
        "Eur High Yield": "#f18712",  # orange
    }

    for col in series_order:
        if col in yields_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=yields_df["Date"],
                    y=yields_df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=2.2, color=series_colors[col]),
                )
            )

    fig.update_yaxes(
        range=[4.5, 9.5],
        tickvals=[4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
        ticktext=["4.5%", "5.0%", "5.5%", "6.0%", "6.5%", "7.0%", "7.5%", "8.0%", "8.5%", "9.0%", "9.5%"],
        title_text="",
    )

    fig.update_xaxes(
        tickvals=yields_df["Date"],
        ticktext=[
            "Dec-22", "Feb-23", "Apr-23", "Jun-23", "Aug-23", "Oct-23",
            "Dec-23", "Feb-24", "Apr-24", "Jun-24", "Aug-24", "Oct-24",
            "Dec-24", "Feb-25", "Apr-25", "Jun-25", "Aug-25", "Oct-25",
        ],
        tickangle=-45,
    )

    apply_dark_chart_layout(
        fig,
        height=height,
        show_legend=True,
        top_margin=8,
        bottom_margin=52,
        left_margin=52,
        right_margin=8,
    )
    return fig


def build_spread_chart(height: int = 176):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=spread_df["Market"],
            y=spread_df["22Q3"],
            name="22Q3",
            marker=dict(color="#1d4fa1"),
        )
    )

    fig.add_trace(
        go.Bar(
            x=spread_df["Market"],
            y=spread_df["4Q25"],
            name="4Q25",
            marker=dict(color="#7ea4d8"),
        )
    )

    fig.update_layout(
        barmode="group",
        bargap=0.24,
        bargroupgap=0.08,
    )

    fig.update_yaxes(
        title_text="bps",
        tickformat=".0f",
        nticks=5,
        zeroline=False,
    )

    fig.update_xaxes(
        tickangle=-30,
    )

    apply_dark_chart_layout(
        fig,
        height=height,
        show_legend=True,
        top_margin=10,
        bottom_margin=48,
        left_margin=42,
        right_margin=8,
    )
    return fig


def build_pan_europe_chart(height: int = 195):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pe_df["Date"],
            y=pe_df["CRE Unlevered"],
            mode="lines+markers",
            name="CRE Unlevered",
            line=dict(width=1.9, color="#1d4fa1"),
            marker=dict(size=4),
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pe_df["Date"],
            y=pe_df["High Yield Bond"],
            mode="lines+markers",
            name="High Yield Bond",
            line=dict(width=1.9, color="#f18712"),
            marker=dict(size=4),
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pe_df["Date"],
            y=pe_df["LT Corporate"],
            mode="lines+markers",
            name="LT Corporate",
            line=dict(width=1.9, color="#7ea4d8"),
            marker=dict(size=4),
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pe_df["Date"],
            y=pe_df["Spread"],
            mode="lines",
            name="Spread",
            line=dict(width=1.7, dash="dot", color="#14284b"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        yaxis=dict(
            title="Return (%)",
            showgrid=True,
            gridcolor="rgba(157,190,238,0.45)",
            zeroline=False,
            tickfont=dict(size=8, color="#d9e3f2"),
            title_font=dict(size=8, color="#d9e3f2"),
            nticks=5,
        ),
        yaxis2=dict(
            title="Spread",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=8, color="#d9e3f2"),
            title_font=dict(size=8, color="#d9e3f2"),
            nticks=5,
        ),
    )

    fig.update_xaxes(
        tickformat="%b\n%Y",
        nticks=6,
    )

    apply_dark_chart_layout(
        fig,
        height=height,
        show_legend=True,
        top_margin=10,
        bottom_margin=26,
        left_margin=42,
        right_margin=42,
    )
    return fig