import streamlit as st


def apply_global_styles():
    st.markdown(
        """
        <style>
        /* =========================
           Global shell
        ========================== */
        :root {
            --page-bg: #06172f;
            --page-bg-2: #041327;
            --text: #edf3fb;
            --muted: #9bb0c8;
            --blue: #66adff;
            --headline: #eef4fb;

            --page-max: 1500px;

            --headline-size: clamp(2.1rem, 2.9vw, 3.05rem);
            --tagline-size: clamp(1.02rem, 1.34vw, 1.42rem);
            --section-title-size: clamp(1.00rem, 1.08vw, 1.20rem);
            --body-size: clamp(0.88rem, 0.97vw, 1.00rem);
            --small-size: clamp(0.68rem, 0.78vw, 0.76rem);

            --page-pad-x: clamp(0.8rem, 1.2vw, 1.15rem);
            --page-pad-y: clamp(0.28rem, 0.55vw, 0.45rem);

            --rule: rgba(130, 156, 192, 0.14);
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(24, 80, 150, 0.16) 0%, rgba(24, 80, 150, 0.00) 18%),
                linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg-2) 100%);
            color: var(--text);
        }

        #MainMenu,
        header[data-testid="stHeader"],
        footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0.08rem;
        }

        .block-container {
            max-width: var(--page-max);
            padding-top: var(--page-pad-y);
            padding-bottom: var(--page-pad-y);
            padding-left: var(--page-pad-x);
            padding-right: var(--page-pad-x);
        }

        div.block-container > div[data-testid="stVerticalBlock"] {
            gap: 0.16rem;
        }

        div[data-testid="column"] > div {
            gap: 0.12rem;
        }

        .element-container {
            margin-bottom: 0.06rem !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0;
        }

        /* tighten streamlit chart spacing */
        [data-testid="stPlotlyChart"] {
            margin-top: -0.08rem;
            margin-bottom: -0.10rem;
        }

        [data-testid="stPlotlyChart"] > div {
            padding: 0 !important;
        }

        /* =========================
           Landing
        ========================== */
        .landing-wrap {
            min-height: 72vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .landing-card {
            width: min(920px, 100%);
            padding: 2.2rem 2.3rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(10, 27, 52, 0.96), rgba(7, 21, 41, 0.96));
            border: 1px solid rgba(130, 156, 192, 0.16);
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
            text-align: center;
        }

        .landing-kicker {
            color: var(--blue);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            margin-bottom: 0.55rem;
        }

        .landing-title {
            color: #f4f8ff;
            font-size: clamp(2.3rem, 5vw, 4.5rem);
            font-weight: 650;
            line-height: 1.02;
            letter-spacing: -0.02em;
            margin: 0 0 0.45rem 0;
        }

        .landing-subtitle {
            color: var(--muted);
            font-size: clamp(0.95rem, 1.2vw, 1.08rem);
            line-height: 1.42;
            max-width: 720px;
            margin: 0 auto;
        }

        /* =========================
           Generic headers
        ========================== */
        .page-header {
            display: flex;
            flex-direction: column;
            gap: 0.08rem;
            margin-bottom: 0.12rem;
        }

        .page-title {
            margin: 0;
            color: #f4f8ff;
            font-size: clamp(1.2rem, 1.7vw, 1.55rem);
            font-weight: 620;
            line-height: 1.06;
            letter-spacing: -0.015em;
        }

        .page-subtitle {
            margin: 0;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.26;
        }

        /* =========================
           Contents
        ========================== */
        .contents-intro {
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.30rem;
        }

        .contents-card {
            min-height: 114px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0.76rem 0.80rem 0.70rem 0.80rem;
            border-radius: 12px;
            border: 1px solid rgba(130, 156, 192, 0.16);
            background: linear-gradient(180deg, rgba(11, 28, 54, 0.95), rgba(7, 21, 41, 0.95));
        }

        .contents-card-title {
            color: #eef4ff;
            font-size: 0.96rem;
            font-weight: 600;
            line-height: 1.14;
            margin: 0 0 0.28rem 0;
        }

        .contents-card-subtitle {
            color: var(--muted);
            font-size: 0.77rem;
            line-height: 1.28;
            margin: 0;
        }

        .contents-card-status {
            color: var(--blue);
            font-size: 0.71rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-top: 0.45rem;
        }

        /* =========================
           Buttons / nav
        ========================== */
        div.stButton > button {
            width: 100%;
            min-height: 2.00rem;
            padding: 0.40rem 0.58rem;
            border-radius: 10px;
            border: 1px solid rgba(130, 156, 192, 0.22);
            background: linear-gradient(180deg, rgba(16, 37, 70, 0.98), rgba(10, 24, 46, 0.98));
            color: #eef4ff;
            font-size: 0.80rem;
            font-weight: 560;
            box-shadow: none;
        }

        div.stButton > button:hover {
            border-color: rgba(102, 173, 255, 0.38);
            color: #ffffff;
        }

        div.stButton > button:focus,
        div.stButton > button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 0.14rem rgba(102, 173, 255, 0.16);
        }

        .nav-strip {
            margin: 0.04rem 0 0.22rem 0;
            padding: 0.28rem 0.40rem 0.26rem 0.40rem;
            border-radius: 10px;
            border: 1px solid rgba(130, 156, 192, 0.12);
            background: rgba(7, 18, 38, 0.54);
        }

        .nav-caption {
            color: var(--muted);
            font-size: 0.70rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 0 0 0.18rem 0;
        }

        /* =========================
           Editorial page layout
        ========================== */
        .layout-2ct {
            display: flex;
            flex-direction: column;
            gap: 0.18rem;
        }

        .layout-2ct-head {
            margin-bottom: 0.03rem;
        }

        .layout-2ct-headline {
            margin: 0;
            color: var(--headline);
            font-size: var(--headline-size);
            font-weight: 640;
            line-height: 1.02;
            letter-spacing: -0.028em;
            max-width: 1320px;
        }

        .layout-2ct-tagline {
            margin: 0.10rem 0 0 0;
            color: var(--blue);
            font-size: var(--tagline-size);
            font-weight: 500;
            line-height: 1.12;
            max-width: 1140px;
        }

        .layout-2ct-left,
        .layout-2ct-right {
            display: flex;
            flex-direction: column;
            gap: 0.12rem;
            min-width: 0;
        }

        /* flatter editorial blocks, not dashboard cards */
        .text-block {
            padding: 0.12rem 0 0.06rem 0;
            border: none;
            background: transparent;
            box-shadow: none;
        }

        .text-block-title {
            margin: 0 0 0.04rem 0;
            color: #eef4fb;
            font-size: var(--section-title-size);
            font-weight: 630;
            line-height: 1.05;
            letter-spacing: -0.01em;
        }

        .text-block-body {
            margin: 0;
            color: var(--text);
            font-size: var(--body-size);
            line-height: 1.16;
        }

        .text-block-body ul {
            margin: 0.06rem 0 0 1.08rem;
            padding: 0;
        }

        .text-block-body li {
            margin: 0 0 0.10rem 0;
            color: var(--text);
            line-height: 1.10;
        }

        .text-block-body strong {
            color: #f5f8fe;
            font-weight: 650;
            font-style: italic;
        }

        .chart-block {
            padding: 0.00rem 0 0 0;
            border: none;
            background: transparent;
            box-shadow: none;
            overflow: visible;
        }

        .chart-block.chart-block-top {
            margin-bottom: 0.02rem;
        }

        .chart-block.chart-block-bottom {
            margin-top: -0.02rem;
        }

        .chart-block-title {
            color: #eef4fb;
            font-size: 0.80rem;
            font-weight: 620;
            line-height: 1.05;
            margin: 0 0 0.04rem 0;
        }

        .chart-block-subtitle {
            color: var(--muted);
            font-size: 0.67rem;
            line-height: 1.12;
            margin: 0 0 0.04rem 0;
        }

        /* =========================
           Tables
        ========================== */
        .compact-table-wrap {
            display: flex;
            flex-direction: column;
            gap: 0.04rem;
        }

        .compact-table-title {
            color: #eef4fb;
            font-size: 0.74rem;
            font-weight: 620;
            line-height: 1.05;
            margin: 0;
        }

        .compact-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.66rem;
        }

        .compact-table thead th {
            background: rgba(37, 69, 115, 0.76);
            color: #eef4ff;
            font-weight: 610;
            padding: 0.16rem 0.24rem;
            text-align: right;
            border: none;
        }

        .compact-table thead th:first-child,
        .compact-table tbody td:first-child {
            text-align: left;
        }

        .compact-table tbody td {
            color: #d9e4f4;
            padding: 0.12rem 0.24rem;
            text-align: right;
            line-height: 1.04;
            border-bottom: 1px solid rgba(130, 156, 192, 0.08);
            word-break: break-word;
        }

        .compact-table tbody tr:last-child td {
            border-bottom: none;
        }

        .compact-table tbody tr:nth-child(even) td {
            background: rgba(255,255,255,0.02);
        }

        /* =========================
           Footer
        ========================== */
        .layout-footer {
            display: flex;
            flex-direction: column;
            gap: 0.02rem;
            margin-top: 0.02rem;
        }

        .source-line {
            padding-top: 0.08rem;
            border-top: 1px solid var(--rule);
            color: var(--muted);
            font-size: var(--small-size);
            line-height: 1.04;
        }

        .disclaimer-line {
            color: var(--muted);
            font-size: var(--small-size);
            line-height: 1.03;
        }

        /* =========================
           Responsive scaling
        ========================== */
        @media (max-width: 1360px) {
            :root {
                --headline-size: clamp(1.85rem, 2.55vw, 2.55rem);
                --tagline-size: clamp(0.94rem, 1.14vw, 1.16rem);
                --section-title-size: clamp(0.92rem, 0.98vw, 1.06rem);
                --body-size: clamp(0.84rem, 0.90vw, 0.93rem);
            }

            .compact-table {
                font-size: 0.62rem;
            }
        }

        @media (max-width: 1120px) {
            :root {
                --headline-size: clamp(1.60rem, 2.15vw, 2.18rem);
                --tagline-size: clamp(0.88rem, 0.98vw, 1.02rem);
                --section-title-size: 0.94rem;
                --body-size: 0.84rem;
            }

            .block-container {
                padding-left: 0.72rem;
                padding-right: 0.72rem;
            }

            .compact-table {
                font-size: 0.59rem;
            }
        }

        @media (max-width: 900px) {
            .layout-2ct-headline,
            .layout-2ct-tagline {
                max-width: 100%;
            }
        }

        @media (max-width: 640px) {
            :root {
                --headline-size: 1.30rem;
                --tagline-size: 0.84rem;
                --section-title-size: 0.88rem;
                --body-size: 0.80rem;
            }

            .block-container {
                padding-left: 0.60rem;
                padding-right: 0.60rem;
            }

            .compact-table {
                font-size: 0.56rem;
            }

            .compact-table thead th,
            .compact-table tbody td {
                padding: 0.12rem 0.18rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )