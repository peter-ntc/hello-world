import streamlit as st
import pandas as pd


def set_page(page_name: str):
    st.session_state["page"] = page_name
    st.rerun()


def set_macro_subpage(page_num: int):
    st.session_state["page"] = "macro"
    st.session_state["macro_subpage"] = page_num
    st.rerun()


def render_header(title: str, subtitle: str = ""):
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    html = f"""<div class="page-header">
<h1 class="page-title">{title}</h1>
{subtitle_html}
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _format_cell_value(val):
    if isinstance(val, float):
        return f"{val:.1f}"
    return val


def render_compact_table(df: pd.DataFrame, title: str = ""):
    title_html = f'<div class="compact-table-title">{title}</div>' if title else ""

    header_cells = "".join(f"<th>{col}</th>" for col in df.columns)

    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_format_cell_value(val)}</td>" for val in row)
        body_rows.append(f"<tr>{cells}</tr>")

    body_html = "".join(body_rows)

    html = f"""<div class="compact-table-wrap">
{title_html}
<table class="compact-table">
<thead>
<tr>{header_cells}</tr>
</thead>
<tbody>
{body_html}
</tbody>
</table>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_text_block(title: str, body: str):
    html = f"""<div class="text-block">
<div class="text-block-title">{title}</div>
<div class="text-block-body">{body}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _open_chart_block(title: str = "", subtitle: str = "", block_class: str = ""):
    title_html = f'<div class="chart-block-title">{title}</div>' if title else ""
    subtitle_html = (
        f'<div class="chart-block-subtitle">{subtitle}</div>' if subtitle else ""
    )
    class_suffix = f" {block_class}" if block_class else ""

    html = f"""<div class="chart-block{class_suffix}">
{title_html}
{subtitle_html}"""
    st.markdown(html, unsafe_allow_html=True)


def _close_chart_block():
    st.markdown("</div>", unsafe_allow_html=True)


def render_layout_2chartandtext(
    headline: str,
    tagline: str,
    left_sections: list = None,
    text_sections: list = None,
    top_content_fn=None,
    bottom_content_fn=None,
    source_text: str = "",
    disclaimer_text: str = "",
    top_title: str = "",
    top_subtitle: str = "",
    bottom_title: str = "",
    bottom_subtitle: str = "",
    left_ratio: float = 0.47,
    right_ratio: float = 0.53,
):
    sections = left_sections if left_sections is not None else text_sections
    if sections is None:
        sections = []

    ratio_sum = left_ratio + right_ratio
    if ratio_sum <= 0:
        left_ratio, right_ratio = 0.47, 0.53
    else:
        left_ratio = left_ratio / ratio_sum
        right_ratio = right_ratio / ratio_sum

    st.markdown('<div class="layout-2ct">', unsafe_allow_html=True)

    head_html = f"""<div class="layout-2ct-head">
<h2 class="layout-2ct-headline">{headline}</h2>
<p class="layout-2ct-tagline">{tagline}</p>
</div>"""
    st.markdown(head_html, unsafe_allow_html=True)

    col_left, col_right = st.columns([left_ratio, right_ratio], gap="small")

    with col_left:
        st.markdown('<div class="layout-2ct-left">', unsafe_allow_html=True)
        for section in sections:
            _render_text_block(section["title"], section["body"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="layout-2ct-right">', unsafe_allow_html=True)

        _open_chart_block(top_title, top_subtitle, block_class="chart-block-top")
        if top_content_fn:
            top_content_fn()
        _close_chart_block()

        _open_chart_block(
            bottom_title, bottom_subtitle, block_class="chart-block-bottom"
        )
        if bottom_content_fn:
            bottom_content_fn()
        _close_chart_block()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="layout-footer">', unsafe_allow_html=True)

    if source_text:
        st.markdown(
            f'<div class="source-line"><strong>Source:</strong> {source_text}</div>',
            unsafe_allow_html=True,
        )

    if disclaimer_text:
        st.markdown(
            f'<div class="disclaimer-line">{disclaimer_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)