# app_tecnologo_streamlit.py — INEP Tecnológico (Brasil/RJ/IDT‑FGV)
# Streamlit app that reads "dados-tecnologo-brasil-rio-idtfgv.csv"
# Tabs:
#   (1) Rio de Janeiro: stacked bar (Presencial x EAD) + line (IDT‑FGV vs RJ)
#   (2) Brasil: stacked bar (Presencial x EAD) + line (Brasil vs RJ)
#
# To run locally:
#   pip install streamlit plotly pandas numpy
#   streamlit run app_tecnologo_streamlit.py

import os
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="INEP • Tecnológico (Brasil/RJ/IDT‑FGV)",
                   page_icon="📊", layout="wide")

# ── STYLE (similar to the reference) ──────────────────────────────────────────
PALETTE = {
    "azuis": ["#002D4D", "#003A79", "#0C63AA", "#008BC9", "#73BFE8"],
    "cinzas": ["#88868B", "#AFAEB4", "#D7D9DD"],
    "verde_claro": "#00E1AC",
    "azul_neon": "#01FFFF",
}
TITLE_COLOR, SUBTITLE_COLOR, GRID_COLOR, AXIS_COLOR = (
    PALETTE["azuis"][0],
    PALETTE["cinzas"][0],
    PALETTE["cinzas"][2],
    PALETTE["cinzas"][1],
)
FONT_FAMILY = "Arial"

COLOR_MOD = {
    "Presencial": PALETTE["azuis"][1],
    "EAD": PALETTE["azuis"][3],
    "Outro": PALETTE["cinzas"][2],
}
COLOR_SERIES = {
    "Rio de Janeiro (Total)": PALETTE["azuis"][1],
    "Brasil (Total)": PALETTE["azuis"][0],
    "IDT‑FGV": PALETTE["azuis"][3],
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def format_short(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return ""
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}k"
    return f"{int(n):d}"


def _norm(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="pages/dados-tecnologo-brasil-rio-idtfgv.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"`{path}` não encontrado no diretório do app.")
        return pd.DataFrame()

    df = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False)

    # Padroniza colunas esperadas
    # Colunas do arquivo: Ano, Instituições, Cursos, Vagas,
    # Ingressantes, Matrículas, Concluintes, Modalidade, Escopo, Modalidade.1
    # Criar MOD2 que mapeia 'A distância' -> 'EAD'
    if "Modalidade" in df.columns:
        df["MOD2"] = df["Modalidade"].replace({"A distância": "EAD"})
    else:
        df["MOD2"] = "Outro"

    # Garante tipos numéricos
    for col in ["Ano", "Vagas", "Ingressantes", "Matrículas", "Concluintes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Ano"])
    df["Ano"] = df["Ano"].astype(int)
    return df


master = load_data()
if master.empty:
    st.stop()

yr_min, yr_max = int(master["Ano"].min()), int(master["Ano"].max())

# ── UI (globais) ──────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:{TITLE_COLOR};font-family:{FONT_FAMILY};font-weight:700;margin-bottom:0'>"
    f"Ensino Tecnológico — INEP {yr_min}–{yr_max}</h2>"
    f"<p style='color:{SUBTITLE_COLOR};font-family:{FONT_FAMILY};margin-top:2px'>"
    f"Comparativos por Modalidade (Presencial × EAD) • Séries: Rio de Janeiro, Brasil e IDT‑FGV • "
    f"Métricas: Ingressantes, Matrículas, Concluintes</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Parâmetros")
    anos_sel = st.slider("Período", min_value=yr_min, max_value=yr_max,
                         value=(yr_min, yr_max), step=1)
    metrica = st.radio("Métrica", ["Matrículas", "Concluintes", "Ingressantes"], index=0, horizontal=False)
    show_labels = st.checkbox("Mostrar rótulos nas barras/linhas", value=True)

yr0, yr1 = anos_sel
metric_map = {"Matrículas": "Matrículas", "Concluintes": "Concluintes", "Ingressantes": "Ingressantes"}
metric_col = metric_map[metrica]
m = master[(master["Ano"] >= yr0) & (master["Ano"] <= yr1)].copy()


# ── COMPONENTS ────────────────────────────────────────────────────────────────
def stacked_by_scope(df, scope_label, title_suffix):
    df_sc = (
        df.loc[df["Escopo"] == scope_label, ["Ano", "MOD2", metric_col]]
        .groupby(["Ano", "MOD2"], as_index=False, observed=True)
        .sum(min_count=1)
    )
    pivot = df_sc.pivot(index="Ano", columns="MOD2", values=metric_col).fillna(0).sort_index()
    for col in ["Presencial", "EAD"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Presencial", "EAD"]]

    fig = go.Figure()
    for mod in ["Presencial", "EAD"]:
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[mod],
                name=mod,
                marker_color=COLOR_MOD.get(mod, PALETTE["azuis"][2]),
                text=[format_short(v) if show_labels else "" for v in pivot[mod]],
                textposition="inside" if show_labels else "none",
                hovertemplate=f"Modalidade: {mod}<br>Ano: "+"%{x}" + "<br>Valor: %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>{metrica} em {yr0}–{yr1}</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>"
                 f"Totais em {title_suffix} (empilhado)</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=110, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     linecolor=AXIS_COLOR, title=metrica, tickformat=",.0f", rangemode="tozero")
    return fig


def line_two_series(df, label_a, label_b, title_suffix):
    # Totais por ano para dois escopos
    a = (
        df.loc[df["Escopo"] == label_a, ["Ano", metric_col]]
        .groupby("Ano", as_index=False)
        .sum(min_count=1)
        .rename(columns={metric_col: "valor"})
    )
    a["serie"] = f"{label_a} (Total)"

    b = (
        df.loc[df["Escopo"] == label_b, ["Ano", metric_col]]
        .groupby("Ano", as_index=False)
        .sum(min_count=1)
        .rename(columns={metric_col: "valor"})
    )
    b["serie"] = f"{label_b} (Total)"

    ts = pd.concat([a, b], ignore_index=True)

    fig = go.Figure()
    for key, sub in ts.groupby("serie"):
        sub = sub.sort_values("Ano")
        fig.add_trace(
            go.Scatter(
                x=sub["Ano"],
                y=sub["valor"],
                mode="lines+markers" + ("+text" if show_labels else ""),
                name=str(key),
                text=[format_short(v) for v in sub["valor"]] if show_labels else None,
                textposition="top center",
                textfont=dict(family=FONT_FAMILY, size=12),
                line=dict(color=COLOR_SERIES.get(str(key), PALETTE["azuis"][2]), width=3),
                marker=dict(size=7),
                cliponaxis=False,
                hovertemplate=f"Série: {key}<br>Ano: "+"%{x}" + "<br>Valor: %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>{metrica} — comparação</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>"
                 f"{title_suffix} • {yr0}–{yr1}</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90,
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=95, b=70),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     linecolor=AXIS_COLOR, title=metrica, tickformat=",.0f", rangemode="tozero")
    return fig


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_rj, tab_br = st.tabs(["🏷️ Rio de Janeiro", "🇧🇷 Brasil"])

with tab_rj:
    st.markdown("### Rio de Janeiro — Presencial × EAD (empilhado)")
    st.plotly_chart(stacked_by_scope(m, "Rio de Janeiro", "Rio de Janeiro"), use_container_width=True)

    # Linha: IDT‑FGV vs RJ
    st.markdown("### Séries temporais — IDT‑FGV × Rio de Janeiro (total)")
    # Monta linha manual (IDT‑FGV vs RJ) pois não é o mesmo helper de dois 'Totals'
    series = []
    rj_total = (
        m.loc[m["Escopo"] == "Rio de Janeiro", ["Ano", metric_col]]
        .groupby("Ano", as_index=False)
        .sum(min_count=1)
        .rename(columns={metric_col: "valor"})
    )
    rj_total["serie"] = "Rio de Janeiro (Total)"
    series.append(rj_total)

    idt = (
        m.loc[m["Escopo"] == "IDT-FGV", ["Ano", metric_col]]
        .groupby("Ano", as_index=False)
        .sum(min_count=1)
        .rename(columns={metric_col: "valor"})
    )
    idt["serie"] = "IDT‑FGV"
    series.append(idt)
    ts = pd.concat(series, ignore_index=True)

    fig_line_rj = go.Figure()
    for key, sub in ts.groupby("serie"):
        sub = sub.sort_values("Ano")
        fig_line_rj.add_trace(
            go.Scatter(
                x=sub["Ano"],
                y=sub["valor"],
                mode="lines+markers" + ("+text" if show_labels else ""),
                name=str(key),
                text=[format_short(v) for v in sub["valor"]] if show_labels else None,
                textposition="top center",
                textfont=dict(family=FONT_FAMILY, size=12),
                line=dict(color=COLOR_SERIES.get(str(key), PALETTE["azuis"][2]), width=3),
                marker=dict(size=7),
                cliponaxis=False,
                hovertemplate=f"Série: {key}<br>Ano: "+"%{x}" + "<br>Valor: %{y:,.0f}<extra></extra>",
            )
        )
    fig_line_rj.update_layout(
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>{metrica} — comparação</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>"
                 f"Somas anuais: IDT‑FGV vs Rio de Janeiro • {yr0}–{yr1}</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90,
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=95, b=70),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
    )
    fig_line_rj.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig_line_rj.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                             linecolor=AXIS_COLOR, title=metrica, tickformat=",.0f", rangemode="tozero")
    st.plotly_chart(fig_line_rj, use_container_width=True)


with tab_br:
    st.markdown("### Brasil — Presencial × EAD (empilhado)")
    st.plotly_chart(stacked_by_scope(m, "Brasil", "Brasil"), use_container_width=True)

    st.markdown("### Séries temporais — Brasil × Rio de Janeiro (total)")
    st.plotly_chart(
        line_two_series(m, "Brasil", "Rio de Janeiro", "Somas anuais: Brasil vs Rio de Janeiro"),
        use_container_width=True,
    )

# ── FOOTER / NOTES ────────────────────────────────────────────────────────────
st.caption("Fonte: INEP • Arquivo: dados-tecnologo-brasil-rio-idtfgv.csv • Modalidade 'A distância' exibida como 'EAD'.")
