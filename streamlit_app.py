import time
import streamlit as st

from styles import apply_global_styles
from helpers import (
    set_page,
    set_macro_subpage,
    render_header,
    render_compact_table,
    render_layout_2chartandtext,
)
from macro_data import gdp_major, gdp_selected
from macro_charts import (
    build_yield_chart,
    build_spread_chart,
    build_pan_europe_chart,
)


st.set_page_config(
    page_title="Townsend View of the World",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if "macro_subpage" not in st.session_state:
    st.session_state["macro_subpage"] = 1

apply_global_styles()


def render_landing():
    st.markdown(
        """
        <div class="landing-wrap">
            <div class="landing-card">
                <div class="landing-kicker">Townsend Research</div>
                <div class="landing-title">View of the World</div>
                <div class="landing-subtitle">
                    An interactive institutional research experience translating Townsend’s
                    editorial views into a more dynamic, structured format.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(3.5)
    set_page("contents")


def render_contents():
    render_header(
        "Contents",
        "Navigate the current V1 sections. Additional sections will be added to this structure over time.",
    )

    st.markdown(
        '<div class="contents-intro">Current prototype coverage includes the landing page, contents page, and Macro pages 1–2.</div>',
        unsafe_allow_html=True,
    )

    cards = [
        ("Macro", "Pages 4–5", True),
        ("RE Cycle", "Pages 6, 8", False),
        ("RE Investment Themes", "Page 9", False),
        ("Capital Market Themes", "Pages 11–12", False),
        ("Global Region / Sector Heat Map", "Page 13", False),
        ("U.S.", "Pages 15–26", False),
        ("Europe", "Pages 28–35", False),
        ("APAC", "Pages 37–44", False),
    ]

    for row_start in range(0, len(cards), 4):
        row_cards = cards[row_start : row_start + 4]
        cols = st.columns(4, gap="small")

        for col, (title, subtitle, is_live) in zip(cols, row_cards):
            with col:
                st.markdown(
                    f"""
                    <div class="contents-card">
                        <div>
                            <div class="contents-card-title">{title}</div>
                            <div class="contents-card-subtitle">{subtitle}</div>
                        </div>
                        <div class="contents-card-status">
                            {"Available now" if is_live else "Coming later"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_live:
                    if st.button(f"Open {title}", key=f"open_{title}"):
                        set_page("macro")
                else:
                    st.button(
                        f"{title} unavailable",
                        key=f"disabled_{title}",
                        disabled=True,
                    )


def render_macro_nav():
    st.markdown('<div class="nav-strip">', unsafe_allow_html=True)
    st.markdown('<div class="nav-caption">Macro section</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.12, 0.82, 0.82, 7.9], gap="small")

    with c1:
        if st.button("← Contents", key="macro_contents_btn"):
            set_page("contents")

    with c2:
        if st.button("Page 1", key="macro_page_1_btn"):
            set_macro_subpage(1)

    with c3:
        if st.button("Page 2", key="macro_page_2_btn"):
            set_macro_subpage(2)

    with c4:
        st.markdown("")

    st.markdown("</div>", unsafe_allow_html=True)


def render_macro_page_1():
    left_sections = [
        {
            "title": "GDP Growth Anticipated to be Healthy",
            "body": """
            <ul>
                <li><strong>US:</strong> Growth resilient; strong labor markets, solid consumption from high-income households (K-shaped economy), and reshoring-linked investment.</li>
                <li><strong>Europe:</strong> Growth stabilizing; real incomes improving and manufacturing indicators turning up.</li>
                <li><strong>APAC:</strong> Australia and Japan both seeing stabilizing to improving outlook.</li>
            </ul>
            """,
        },
        {
            "title": "Inflation Coming Under Control",
            "body": """
            <ul>
                <li>In the US, Europe, and Australia, disinflation is broad-based across goods, housing, and core services.</li>
                <li>Japan is the outlier with wage-gain-driven inflation.</li>
            </ul>
            """,
        },
        {
            "title": "Interest Rates Declining and Likely to Remain Below Recent Highs",
            "body": """
            <ul>
                <li><strong>US:</strong> Fed easing underway, BBB bond yields (proxy for core borrowing cost) and high yields (proxy for non-core borrowing cost) declining.</li>
                <li><strong>Europe:</strong> ECB cutting; lower sovereign yields reducing financing costs and supporting cap-rate compression.</li>
                <li><strong>Australia:</strong> Anticipated rate cuts as inflation moderates; lending spreads narrowing and refinancing improving.</li>
                <li><strong>Japan:</strong> BOJ policy steady with mild normalization; rates remain lowest globally and unlikely to fall further.</li>
            </ul>
            """,
        },
    ]

    def render_top():
        t1, t2 = st.columns([1, 1], gap="small")
        with t1:
            render_compact_table(gdp_major, "")
        with t2:
            render_compact_table(gdp_selected, "")

    def render_bottom():
        st.plotly_chart(
            build_yield_chart(height=210),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    render_layout_2chartandtext(
        headline="Lower Rates are Catalyzing a Turn in Real Estate Valuations",
        tagline="Lower borrowing costs and easing cap rates are providing broad support to valuations",
        left_sections=left_sections,
        top_content_fn=render_top,
        bottom_content_fn=render_bottom,
        top_title="Real GDP Forecasts (YoY%)",
        top_subtitle="",
        bottom_title="High Yield Bond Effective Yields",
        bottom_subtitle="",
        source_text="Bloomberg (January 2026), Federal Reserve Bank of St. Louis (December 2025).",
        disclaimer_text="Townsend’s views are as of the date of this publication and may be changed or modified at any time and without notice. Past performance is not indicative of future results. Actual results and developments may differ materially from those expressed or implied herein.",
        left_ratio=0.49,
        right_ratio=0.51,
    )


def render_macro_page_2():
    left_sections = [
        {
            "title": "Valuation Signals Point to Opportunity",
            "body": """
            <ul>
                <li><strong>Wide yield spreads:</strong> Cap rates remain well above government bond yields across major markets, with spreads widening over the past year as rates fell and property yields rose.</li>
                <li><strong>Private-market spreads attractive:</strong> In the U.S., risk-adjusted private real estate returns (~7–8%) exceed both Baa and high-yield bond benchmarks.</li>
                <li><strong>Strong corporate-bond comparison:</strong> CRE spreads versus investment-grade bonds remain above long-term norms, a historically supportive valuation signal.</li>
                <li><strong>Public market confirmation:</strong> U.S. REITs trade cheaply relative to the S&amp;P 500, with AFFO yields above equity earnings yields—supporting valuation upside.</li>
                <li><strong>Europe also compelling:</strong> Pan-European real estate shows elevated spreads versus corporate bonds, with fair-value indicators still in “cheap” territory.</li>
                <li><strong>Cycle inflection:</strong> Historically, wide CRE–bond spreads have preceded periods of above-average forward returns.</li>
            </ul>
            """,
        },
    ]

    def render_top():
        st.plotly_chart(
            build_spread_chart(height=205),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    def render_bottom():
        st.plotly_chart(
            build_pan_europe_chart(height=215),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    render_layout_2chartandtext(
        headline="Yield Spreads Signal Real Estate is Attractively Priced at This Point",
        tagline="Falling rates and rising property yields have opened a valuation gap supportive of forward returns",
        left_sections=left_sections,
        top_content_fn=render_top,
        bottom_content_fn=render_bottom,
        top_title="10-yr Government Bond Yields vs. Average Cap Rate (Office, Industrial, Retail) Spreads",
        top_subtitle="",
        bottom_title="Pan-European RE Unlevered Performance Forecasts vs. Income",
        bottom_subtitle="",
        source_text="MSCI Real Capital Analytics (Q4 2025), Green Street (December 2025).",
        disclaimer_text="Townsend’s views are as of the date of this presentation and may be changed or modified at any time and without notice. Past performance is not indicative of future results. Actual results and developments may differ materially from those expressed or implied herein.",
        left_ratio=0.47,
        right_ratio=0.53,
    )


def render_macro():
    render_macro_nav()

    if st.session_state.get("macro_subpage", 1) == 1:
        render_macro_page_1()
    else:
        render_macro_page_2()


page = st.session_state.get("page", "landing")

if page == "landing":
    render_landing()
elif page == "contents":
    render_contents()
elif page == "macro":
    render_macro()
else:
    render_contents()