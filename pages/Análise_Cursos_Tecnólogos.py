# app_tecnologo_streamlit.py — INEP Tecnológico (Brasil/RJ/IDT-FGV)
# Streamlit app that reads "dados-tecnologo-brasil-rio-idtfgv.csv"
# Tabs:
#   (1) Rio de Janeiro: stacked bar (Presencial x EAD) + line (IDT-FGV vs RJ) + 100% stacked (market share IDT-FGV)
#   (2) Brasil: stacked bar (Presencial x EAD) + line (Brasil vs RJ) + Treemap IES (UF → IES) com toggle e ano
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
st.set_page_config(page_title="INEP • Tecnológico (Brasil/RJ/IDT-FGV)",
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
    "IDT-FGV": PALETTE["azuis"][3],
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

# ── EXTRA DATA (IES aggregated for Treemap) ───────────────────────────────────
@st.cache_data
def load_ies_agg(path="pages/dados-agrupados-tecnologos-por-faculdade-ano.csv") -> pd.DataFrame:
    """
    Expected columns:
      - NU_ANO_CENSO (int)
      - SG_UF_IES (str), CO_IES (id), NO_IES (str)
      - QT_MAT, QT_ING, QT_CONC (numeric)
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    ies = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False)
    ies["NU_ANO_CENSO"] = pd.to_numeric(ies.get("NU_ANO_CENSO"), errors="coerce").astype("Int64")
    for c in ["QT_MAT", "QT_ING", "QT_CONC"]:
        if c in ies.columns:
            ies[c] = pd.to_numeric(ies[c], errors="coerce")
    return ies

master = load_data()
if master.empty:
    st.stop()

yr_min, yr_max = int(master["Ano"].min()), int(master["Ano"].max())

# ── UI (globais) ──────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:{TITLE_COLOR};font-family:{FONT_FAMILY};font-weight:700;margin-bottom:0'>"
    f"Ensino Tecnológico — INEP {yr_min}–{yr_max}</h2>"
    f"<p style='color:{SUBTITLE_COLOR};font-family:{FONT_FAMILY};margin-top:2px'>"
    f"Comparativos por Modalidade (Presencial × EAD) • Séries: Rio de Janeiro, Brasil e IDT-FGV • "
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
    pivot = pivot["Presencial"].to_frame().join(pivot["EAD"].to_frame(), how="outer").fillna(0)

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
    idt["serie"] = "IDT-FGV"
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

    # 100% stacked bar — Market share da IDT‑FGV no RJ
    st.markdown("### Market share — IDT‑FGV no RJ (100%)")
    rj_tot = (
        m.loc[m["Escopo"] == "Rio de Janeiro", ["Ano", metric_col]]
         .groupby("Ano", as_index=False).sum(min_count=1)
         .rename(columns={metric_col: "RJ"})
    )
    idt_tot = (
        m.loc[m["Escopo"] == "IDT-FGV", ["Ano", metric_col]]
         .groupby("Ano", as_index=False).sum(min_count=1)
         .rename(columns={metric_col: "IDT_FGV"})
    )
    share = rj_tot.merge(idt_tot, on="Ano", how="left").fillna({"IDT_FGV": 0})
    share["OUTROS_RJ"] = (share["RJ"] - share["IDT_FGV"]).clip(lower=0)

    denom = share["RJ"].replace(0, np.nan)
    pct_idt = (share["IDT_FGV"] / denom * 100).fillna(0)
    pct_out = (share["OUTROS_RJ"] / denom * 100).fillna(0)

    fig_share = go.Figure()
    fig_share.add_trace(
        go.Bar(
            x=share["Ano"],
            y=pct_out,
            name="Demais RJ",
            marker_color=PALETTE["cinzas"][1],
            text=[f"{v:.1f}%" if show_labels else "" for v in pct_out],
            textposition="inside" if show_labels else "none",
            hovertemplate="Ano: %{x}<br>Demais RJ: %{y:.1f}%<br>Absoluto: %{customdata:,}<extra></extra>",
            customdata=share["OUTROS_RJ"],
        )
    )
    fig_share.add_trace(
        go.Bar(
            x=share["Ano"],
            y=pct_idt,
            name="IDT‑FGV",
            marker_color=PALETTE["azuis"][3],
            text=[f"{v:.1f}%" if show_labels else "" for v in pct_idt],
            textposition="inside" if show_labels else "none",
            hovertemplate="Ano: %{x}<br>IDT‑FGV: %{y:.1f}%<br>Absoluto: %{customdata:,}<extra></extra>",
            customdata=share["IDT_FGV"],
        )
    )
    fig_share.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>Participação (%) — IDT‑FGV no RJ</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>"
                 f"Base: totais anuais do Rio de Janeiro • {yr0}–{yr1}</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=110, b=60),
    )
    fig_share.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig_share.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                           linecolor=AXIS_COLOR, title="Participação (%)",
                           tickformat=".0f", range=[0, 100])
    st.plotly_chart(fig_share, use_container_width=True)


with tab_br:
    st.markdown("### Brasil — Presencial × EAD (empilhado)")
    st.plotly_chart(stacked_by_scope(m, "Brasil", "Brasil"), use_container_width=True)

    st.markdown("### Séries temporais — Brasil × Rio de Janeiro (total)")
    st.plotly_chart(
        line_two_series(m, "Brasil", "Rio de Janeiro", "Somas anuais: Brasil vs Rio de Janeiro"),
        use_container_width=True,
    )

    # 100% stacked bar — Market share de RJ dentro do Brasil
    st.markdown("### Market share — RJ no Brasil (100%)")
    br_tot = (
        m.loc[m["Escopo"] == "Brasil", ["Ano", metric_col]]
         .groupby("Ano", as_index=False).sum(min_count=1)
         .rename(columns={metric_col: "BR"})
    )
    rj_tot2 = (
        m.loc[m["Escopo"] == "Rio de Janeiro", ["Ano", metric_col]]
         .groupby("Ano", as_index=False).sum(min_count=1)
         .rename(columns={metric_col: "RJ"})
    )
    share_br = br_tot.merge(rj_tot2, on="Ano", how="left").fillna({"RJ": 0})
    share_br["DEMAIS_BR"] = (share_br["BR"] - share_br["RJ"]).clip(lower=0)

    denom_br = share_br["BR"].replace(0, np.nan)
    pct_rj = (share_br["RJ"] / denom_br * 100).fillna(0)
    pct_out_br = (share_br["DEMAIS_BR"] / denom_br * 100).fillna(0)

    fig_share_br = go.Figure()
    fig_share_br.add_trace(
        go.Bar(
            x=share_br["Ano"],
            y=pct_out_br,
            name="Demais Brasil",
            marker_color=PALETTE["cinzas"][1],
            text=[f"{v:.1f}%" if show_labels else "" for v in pct_out_br],
            textposition="inside" if show_labels else "none",
            hovertemplate="Ano: %{x}<br>Demais Brasil: %{y:.1f}%<br>Absoluto: %{customdata:,}<extra></extra>",
            customdata=share_br["DEMAIS_BR"],
        )
    )
    fig_share_br.add_trace(
        go.Bar(
            x=share_br["Ano"],
            y=pct_rj,
            name="Rio de Janeiro",
            marker_color=PALETTE["azuis"][1],
            text=[f"{v:.1f}%" if show_labels else "" for v in pct_rj],
            textposition="inside" if show_labels else "none",
            hovertemplate="Ano: %{x}<br>Rio de Janeiro: %{y:.1f}%<br>Absoluto: %{customdata:,}<extra></extra>",
            customdata=share_br["RJ"],
        )
    )
    fig_share_br.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>Participação (%) — RJ no Brasil</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>"
                 f"Base: totais anuais do Brasil • {yr0}–{yr1}</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=110, b=60),
    )
    fig_share_br.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig_share_br.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                              linecolor=AXIS_COLOR, title="Participação (%)",
                              tickformat=".0f", range=[0, 100])
    st.plotly_chart(fig_share_br, use_container_width=True)

    # Treemap — IES (UF → IES) — com toggle de métrica e slider de ano (2022–2024)
    st.markdown("### Treemap — IES (UF → IES)")
    ies = load_ies_agg()
    if ies.empty:
        st.info("Arquivo 'dados-agrupados-tecnologos-por-faculdade-ano.csv' não encontrado — treemap indisponível.")
    else:
        import plotly.express as px
        cols = st.columns([1, 1])
        with cols[0]:
            metric_choice = st.radio("Métrica do treemap", ["QT_MAT", "QT_ING", "QT_CONC"], index=0, horizontal=True)
        with cols[1]:
            available_years = sorted([int(y) for y in ies["NU_ANO_CENSO"].dropna().unique()])
            if available_years:
                min_avail = max(2020, min(available_years))
                max_avail = min(2024, max(available_years))
                if min_avail > max_avail:
                    st.warning("Não há dados entre 2020 e 2024 para o treemap.")
                    ano_treemap = None
                else:
                    ano_treemap = st.slider("Ano do treemap (2020–2024)", min_value=min_avail, max_value=max_avail, value=max_avail, step=1)
            else:
                st.warning("Arquivo de IES sem anos válidos para treemap.")
                ano_treemap = None

        if ano_treemap is not None:
            num_cols = ["QT_MAT", "QT_ING", "QT_CONC"]
            agg_yr = (
                ies.loc[ies["NU_ANO_CENSO"] == int(ano_treemap)]
                   .groupby(["SG_UF_IES", "CO_IES", "NO_IES"], as_index=False)[num_cols]
                   .sum(min_count=1)
            )

            if agg_yr.empty:
                st.warning(f"Sem dados para o ano {ano_treemap} no arquivo de IES.")
            else:
                values_col = metric_choice
                color_col = metric_choice

                fig_treemap = px.treemap(
                    agg_yr,
                    path=["SG_UF_IES", "NO_IES"],
                    values=values_col,
                    color=color_col,
                    hover_data={
                        "CO_IES": True,
                        "QT_MAT": ":,",
                        "QT_ING": ":,",
                        "QT_CONC": ":,",
                    },
                    title=f"Treemap • IES (UF → IES) • Ano {ano_treemap} • Métrica: {metric_choice}",
                    color_continuous_scale=[PALETTE["azuis"][3], PALETTE["azuis"][1], PALETTE["azuis"][0]],
                )
                fig_treemap.update_traces(textinfo="label+value")
                fig_treemap.update_layout(
                    paper_bgcolor="white",
                    margin=dict(l=10, r=10, t=80, b=10),
                    font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
                )
                st.plotly_chart(fig_treemap, use_container_width=True)

# ── FOOTER / NOTES ────────────────────────────────────────────────────────────
st.caption("Fonte: INEP • Arquivo: dados-tecnologo-brasil-rio-idtfgv.csv; dados-agrupados-tecnologos-por-faculdade-ano.csv • Modalidade 'A distância' exibida como 'EAD'.")
