# pages/01_Analise_Escolas_por_UF.py
# Versão interativa (Plotly) mantendo o design dos charts atuais (Matplotlib)

import os
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ===========================
# DESIGN / PALETA
# ===========================
BLUES = {"navy":"#002D4D", "sky":"#73BFE8", "medium":"#0C63AA"}
GRAYS = {"mid":"#88868B", "light":"#D7D9DD", "white":"#FFFFFF"}
NON_BLUE_HEX = [
    "#BDA132", "#B06F0B", "#7F1343", "#78487F", "#2B8671", "#668040",
    "#A02BAD", "#C680CE", "#782082", "#00E1AC", "#66EDCD", "#00A981",
]
FONT_FAMILY = "Arial"

# “espessuras” para manter destaque
EBAPE_LW, EAESP_LW, OTHERS_LW = 3.2, 2.6, 1.3
MARKER_SIZE = 6

# “espaços” análogos aos knobs anteriores (efeito aproximado em Plotly)
TITLE_Y = 0.97          # sobe/baixa bloco de título+subtítulo
LEGEND_Y = -0.16        # legenda “colada” abaixo do gráfico (negativo = fora do plot)
TITLE_SIZE = 20
SUBTITLE_SIZE = 14
AXIS_TICK_SIZE = 12
LABEL_FS = 12           # rótulos sobre os pontos

# ===========================
# HELPERS
# ===========================
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def _normcol(s: str) -> str:
    s = _strip_accents(s).lower().strip().replace("  ", " ")
    return s.replace("-", "_").replace(" ", "_")

def _read_csv_robusto(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _mapear_colunas(df: pd.DataFrame) -> pd.DataFrame:
    if "IES" in df.columns and not pd.api.types.is_string_dtype(df["IES"]):
        df = df.rename(columns={"IES": "IES_id"})
    alvo_simples = {
        "UF": {"uf","sg_uf","estado"},
        "Ano": {"ano","nu_ano_censo","ano_censo","ano_ref"},
        "Ingresso": {"ingresso","ingressos","ingressantes"},
        "Matriculas": {"matriculas","matriculas_total","matriculas_totais","matriculas_geral","matricula"},
        "Concluintes": {"concluintes","conclusoes","concluinte"},
    }
    nomes_possiveis = ["sigla","ies_sigla","ies_-_sigla","no_ies","nome_ies","ies_nome","instituicao","faculdade"]
    norm2orig = {_normcol(c): c for c in df.columns}
    ren = {}
    for alvo, cands in alvo_simples.items():
        for cand in cands:
            if cand in norm2orig:
                ren[norm2orig[cand]] = alvo; break
    for cand in nomes_possiveis:
        if cand in norm2orig:
            ren[norm2orig[cand]] = "IES"; break
    if not any(v == "IES" for v in ren.values()):
        if "IES_id" in df.columns: ren["IES_id"] = "IES"
    if ren: df = df.rename(columns=ren)
    req = ["UF","IES","Ano","Ingresso","Matriculas","Concluintes"]
    miss = [c for c in req if c not in df.columns]
    if miss: raise KeyError(f"Faltam colunas: {miss}\nDisponíveis: {list(df.columns)}")
    if pd.api.types.is_numeric_dtype(df["IES"]):
        df["IES"] = df["IES"].astype(str)
    return df

def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    def to_num(s):
        if pd.isna(s): return np.nan
        s = str(s).strip().replace(".", "").replace(",", ".")
        try: return float(s)
        except Exception: return pd.to_numeric(s, errors="coerce")
    for col in ["Ingresso","Matriculas","Concluintes"]:
        df[col] = df[col].map(to_num)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["UF"]  = df["UF"].astype(str).str.upper().str.strip()
    df["IES"] = df["IES"].astype(str).str.strip()
    df["IES_norm"] = df["IES"].map(lambda s: _strip_accents(s).lower())
    return df

def is_ebape(name_norm: str) -> bool:
    s = name_norm
    return ("ebape" in s) or ("fgv ebape" in s) or ("escola brasileira de administracao publica" in s)

def is_eaesp(name_norm: str) -> bool:
    s = name_norm
    return ("eaesp" in s) or ("fgv eaesp" in s) or ("escola de administracao de empresas de sao paulo" in s)

def _fmt_int(v):
    try: return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception: return ""

def palette_distinct_others(n: int) -> list:
    base = NON_BLUE_HEX[:]
    if n <= len(base): return base[:n]
    # se faltar cor, pequenas variações suavizadas
    def _mix_with_gray(hex_color: str, alpha: float) -> str:
        R,G,B = 136,136,136
        hc = hex_color.lstrip("#")
        r,g,b = int(hc[0:2],16), int(hc[2:4],16), int(hc[4:6],16)
        r = int(r + (R - r) * alpha); g = int(g + (G - g) * alpha); b = int(b + (B - b) * alpha)
        return f"#{r:02X}{g:02X}{b:02X}"
    i = 0
    while len(base) < n:
        base.append(_mix_with_gray(NON_BLUE_HEX[i % len(NON_BLUE_HEX)], 0.25)); i += 1
    return base[:n]

# ===========================
# DATASET
# ===========================
CSV_PATH = "planilha-escolas-rj-sp-mg-from-powerbi_atualizado_2024.csv"  # hardcoded, igual ao seu código

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df_raw = _read_csv_robusto(CSV_PATH)
    df = _mapear_colunas(df_raw)
    df = _coerce_numerics(df)
    df = (df.groupby(["UF","IES","Ano"], as_index=False)
            .agg({"Ingresso":"sum","Matriculas":"sum","Concluintes":"sum"}))
    df["IES_norm"] = df["IES"].map(lambda s: _strip_accents(s).lower())
    return df

df = load_dataset()

# ===========================
# FIGURA PLOTLY (mantendo o design)
# ===========================
def fig_competicao_plotly(df: pd.DataFrame, uf: str, metric: str) -> go.Figure:
    title_map  = {"Ingresso":"INGRESSO", "Matriculas":"MATRICULADOS", "Concluintes":"CONCLUINTES"}
    assert metric in title_map

    # Subconjunto + EBAPE em todos + EAESP só em SP
    df_uf = df[df["UF"].str.fullmatch(uf, na=False)].copy()
    frames = [df_uf, df[df["IES_norm"].apply(is_ebape)]]
    if uf == "SP":
        frames.append(df[df["IES_norm"].apply(is_eaesp)])
    df_plot = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["UF","IES","Ano","Ingresso","Matriculas","Concluintes"]))

    anos = sorted(df_plot["Ano"].dropna().unique())
    if not anos:
        return go.Figure()

    # Cores por IES
    unique_ies = sorted(df_plot["IES"].unique())
    ebapes = [i for i in unique_ies if is_ebape(_strip_accents(i).lower())]
    eaesps = [i for i in unique_ies if (uf == "SP" and is_eaesp(_strip_accents(i).lower()))] if uf == "SP" else []
    others = [i for i in unique_ies if i not in ebapes + eaesps]

    cmap = {ies: col for ies, col in zip(others, palette_distinct_others(len(others)))}
    for nm in ebapes: cmap[nm] = BLUES["sky"]
    for nm in eaesps: cmap[nm] = BLUES["navy"]

    order = others + eaesps + ebapes  # EBAPE por último (fica por cima)

    # --- Monta figura
    fig = go.Figure()

    # 1) traços de TEXTO primeiro (para ficarem atrás das linhas)
    for ies in order:
        dat = df_plot[df_plot["IES"] == ies][["Ano", metric]].dropna().sort_values("Ano")
        if dat.empty: 
            continue
        xs, ys = dat["Ano"].values, dat[metric].values
        color = cmap.get(ies, GRAYS["mid"])
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="text", name=str(ies), showlegend=False,
                text=[_fmt_int(v) for v in ys],
                textposition="top center",
                textfont=dict(family=FONT_FAMILY, size=LABEL_FS, color=color),
                cliponaxis=False, hoverinfo="skip"
            )
        )

    # 2) traços de LINHA+MARCADOR por cima (mantém “linhas acima dos rótulos”)
    for ies in order:
        dat = df_plot[df_plot["IES"] == ies][["Ano", metric]].dropna().sort_values("Ano")
        if dat.empty: continue
        xs, ys = dat["Ano"].values, dat[metric].values
        n = _strip_accents(ies).lower()
        is_EBAPE = is_ebape(n)
        is_EAESP = (uf == "SP" and is_eaesp(n))
        lw = EBAPE_LW if is_EBAPE else (EAESP_LW if is_EAESP else OTHERS_LW)
        color = cmap.get(ies, GRAYS["mid"])
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name=str(ies),
                line=dict(color=color, width=lw),
                marker=dict(size=MARKER_SIZE, color=color),
                cliponaxis=False,
                hovertemplate=f"IES: {ies}<br>Ano: %{{x}}<br>{title_map[metric]}: %{{y:,.0f}}<extra></extra>",
            )
        )

    # --- Layout: Título + Subtítulo (alinhados à esquerda) + legenda “colada”
    title_html = (
        f"<b style='color:{BLUES['navy']}'>COMPETIÇÃO ENTRE ESCOLAS - {uf}</b>"
        f"<br><span style='color:{BLUES['medium']}; font-weight: normal;'>{title_map[metric]}</span>"
    )
    fig.update_layout(
        title=dict(text=title_html, font=dict(family=FONT_FAMILY, size=TITLE_SIZE), x=0, xanchor="left", y=TITLE_Y),
        legend=dict(orientation="h", yanchor="top", y=LEGEND_Y, xanchor="left", x=0,
                    font=dict(family=FONT_FAMILY, size=12)),
        margin=dict(l=10, r=10, t=90, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=GRAYS["mid"]),
    )
    # Grade só horizontal; eixos “clean”
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRAYS["mid"],
                     tickmode="linear", tickfont=dict(size=AXIS_TICK_SIZE))
    fig.update_yaxes(showgrid=True, gridcolor=GRAYS["light"], zeroline=False,
                     linecolor=GRAYS["mid"], tickformat=",.0f", tickfont=dict(size=AXIS_TICK_SIZE))
    return fig

def fig_marketshare_100_plotly(df: pd.DataFrame, uf: str, metric: str) -> go.Figure:
    """
    Market share das IES por ano (100% stacked), incluindo EBAPE em todos os UFs
    e EAESP apenas em SP, mantendo as mesmas cores e ordem visual.
    """
    title_map  = {"Ingresso":"INGRESSO", "Matriculas":"MATRICULADOS", "Concluintes":"CONCLUINTES"}
    assert metric in title_map

    # Subconjunto por UF + EBAPE (sempre) + EAESP (só em SP)
    df_uf = df[df["UF"].str.fullmatch(uf, na=False)].copy()
    frames = [df_uf, df[df["IES_norm"].apply(is_ebape)]]
    if uf == "SP":
        frames.append(df[df["IES_norm"].apply(is_eaesp)])
    df_plot = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["UF","IES","Ano","Ingresso","Matriculas","Concluintes"]))

    anos = sorted(df_plot["Ano"].dropna().unique())
    if not anos:
        return go.Figure()

    # Paleta igual à do gráfico de linha
    unique_ies = sorted(df_plot["IES"].unique())
    ebapes = [i for i in unique_ies if is_ebape(_strip_accents(i).lower())]
    eaesps = [i for i in unique_ies if (uf == "SP" and is_eaesp(_strip_accents(i).lower()))] if uf == "SP" else []
    others = [i for i in unique_ies if i not in ebapes + eaesps]

    cmap = {ies: col for ies, col in zip(others, palette_distinct_others(len(others)))}
    for nm in ebapes: cmap[nm] = BLUES["sky"]
    for nm in eaesps: cmap[nm] = BLUES["navy"]

    order = others + eaesps + ebapes  # EBAPE por último → faixa superior

    # Pivot para shares (%)
    pivot_abs = (df_plot[["Ano", "IES", metric]]
                 .dropna()
                 .groupby(["Ano", "IES"], as_index=False)[metric]
                 .sum())
    pivot_abs = pivot_abs.pivot(index="Ano", columns="IES", values=metric).reindex(index=anos).fillna(0)

    # garantir todas as IES nas colunas, na ordem desejada
    for ies in order:
        if ies not in pivot_abs.columns:
            pivot_abs[ies] = 0
    pivot_abs = pivot_abs[order]

    totals = pivot_abs.sum(axis=1).replace(0, np.nan)
    pivot_pct = (pivot_abs.div(totals, axis=0) * 100).fillna(0)

    # --- Figura
    fig = go.Figure()

    # Traços em ordem (others → EAESP → EBAPE) para formar o empilhado 100%
    for ies in order:
        fig.add_trace(
            go.Bar(
                x=pivot_pct.index,
                y=pivot_pct[ies].values,
                name=str(ies),
                marker=dict(color=cmap.get(ies, GRAYS["mid"]), line=dict(width=0)),
                # enviamos ABS e % como customdata para usar no hover
                customdata=np.stack([pivot_abs[ies].values, pivot_pct[ies].values], axis=-1),
                hovertemplate=(
                    f"IES: {ies}<br>"
                    "Ano: %{x}<br>"
                    # 👇 repare nas chaves duplas para escapar no f-string
                    f"{title_map[metric]}: %{{customdata[0]:,.0f}}<br>"
                    "Market share: %{{customdata[1]:.1f}}%<extra></extra>"
                ),
            )
        )

    # Layout — alinhado ao design
    title_html = (
        f"<b style='color:{BLUES['navy']}'>COMPETIÇÃO ENTRE ESCOLAS - {uf}</b>"
        f"<br><span style='color:{BLUES['medium']}; font-weight: normal;'>{title_map[metric]} — Market Share 100%</span>"
    )
    fig.update_layout(
        barmode="stack",
        bargap=0.0,          # colunas encostadas
        bargroupgap=0.0,
        title=dict(text=title_html, font=dict(family=FONT_FAMILY, size=TITLE_SIZE), x=0, xanchor="left", y=TITLE_Y),
        legend=dict(orientation="h", yanchor="top", y=LEGEND_Y, xanchor="left", x=0,
                    font=dict(family=FONT_FAMILY, size=12)),
        margin=dict(l=10, r=10, t=90, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=GRAYS["mid"]),
    )
    # Eixos no padrão: grade horizontal apenas, 0–100%
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRAYS["mid"], tickmode="linear")
    fig.update_yaxes(
        showgrid=True, gridcolor=GRAYS["light"], zeroline=False, linecolor=GRAYS["mid"],
        range=[0, 100], tickmode="linear", dtick=20, ticksuffix="%", title="Market share (%)"
    )
    return fig

def _load_raw_with_tipo_sigla() -> pd.DataFrame:
    df_raw = _read_csv_robusto(CSV_PATH)
    df_raw = _mapear_colunas(df_raw)
    df_raw = _coerce_numerics(df_raw)
    return df_raw

def fig_rj_tipo_ebape_plotly(metric: str) -> go.Figure:
    title_map = {"Ingresso":"INGRESSO", "Matriculas":"MATRICULADOS", "Concluintes":"CONCLUINTES"}
    assert metric in title_map

    d = _load_raw_with_tipo_sigla()
    d = d[d["UF"].str.fullmatch("RJ", na=False)].copy()
    if d.empty or metric not in d.columns:
        return go.Figure()

    # Normalizações
    d["Tipo_norm"] = d.get("Tipo", "").astype(str).map(lambda s: _strip_accents(s).lower())
    if "Sigla" in d.columns:
        ebape_mask = d["Sigla"].astype(str).str.upper().str.strip().eq("EBAPE")
    else:
        # fallback robusto caso 'Sigla' não exista
        ebape_mask = d["IES_norm"].apply(is_ebape) if "IES_norm" in d.columns else d["IES"].astype(str).str.contains("EBAPE", case=False, na=False)

    mask_pub  = (~ebape_mask) & d["Tipo_norm"].str.startswith("public")
    mask_priv = (~ebape_mask) & d["Tipo_norm"].str.startswith("privad")

    by_year = {
        "Pública": d.loc[mask_pub,  ["Ano", metric]].dropna().groupby("Ano", as_index=False)[metric].sum(),
        "Privada": d.loc[mask_priv, ["Ano", metric]].dropna().groupby("Ano", as_index=False)[metric].sum(),
        "EBAPE":   d.loc[ebape_mask,["Ano", metric]].dropna().groupby("Ano", as_index=False)[metric].sum(),
    }

    anos = sorted(set().union(*[set(df["Ano"].unique()) for df in by_year.values() if not df.empty]))
    if not anos: 
        return go.Figure()

    color_map = {"Pública": GRAYS["mid"], "Privada": BLUES["navy"], "EBAPE": BLUES["sky"]}
    lw_map    = {"Pública": OTHERS_LW,   "Privada": OTHERS_LW,     "EBAPE": EBAPE_LW}

    fig = go.Figure()

    # Rótulos de texto (atrás das linhas)
    for label in ["Pública", "Privada", "EBAPE"]:
        dfy = by_year[label]
        if dfy.empty: 
            continue
        xs, ys = dfy["Ano"].values, dfy[metric].values
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="text", name=label, showlegend=False,
                text=[_fmt_int(v) for v in ys],
                textposition="top center",
                textfont=dict(family=FONT_FAMILY, size=LABEL_FS, color=color_map[label]),
                cliponaxis=False, hoverinfo="skip"
            )
        )

    # Linhas + marcadores
    for label in ["Pública", "Privada", "EBAPE"]:
        dfy = by_year[label]
        if dfy.empty: 
            continue
        xs, ys = dfy["Ano"].values, dfy[metric].values
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=label,
                line=dict(color=color_map[label], width=lw_map[label]),
                marker=dict(size=MARKER_SIZE, color=color_map[label]),
                cliponaxis=False,
                hovertemplate=f"Categoria: {label}<br>Ano: %{{x}}<br>{title_map[metric]}: %{{y:,.0f}}<extra></extra>",
            )
        )

    title_html = (
        f"<b style='color:{BLUES['navy']}'>TIPO DE IES — RJ</b>"
        f"<br><span style='color:{BLUES['medium']}; font-weight: normal;'>{title_map[metric]} — Públicas vs Privadas vs EBAPE</span>"
    )
    fig.update_layout(
        title=dict(text=title_html, font=dict(family=FONT_FAMILY, size=TITLE_SIZE), x=0, xanchor="left", y=TITLE_Y),
        legend=dict(orientation="h", yanchor="top", y=LEGEND_Y, xanchor="left", x=0,
                    font=dict(family=FONT_FAMILY, size=12)),
        margin=dict(l=10, r=10, t=90, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=GRAYS["mid"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRAYS["mid"],
                     tickmode="linear", tickfont=dict(size=AXIS_TICK_SIZE))
    fig.update_yaxes(showgrid=True, gridcolor=GRAYS["light"], zeroline=False,
                     linecolor=GRAYS["mid"], tickformat=",.0f", tickfont=dict(size=AXIS_TICK_SIZE))
    return fig


# ===========================
# UI
# ===========================
st.title("Análise — Escolas por UF (interativo)")
tabs = st.tabs(["SP", "RJ", "MG"])
for tab, UF in zip(tabs, ["SP", "RJ", "MG"]):
    with tab:
        for MET in ["Ingresso", "Matriculas", "Concluintes"]:
            # 1) Gráfico de linha (igual ao atual)
            fig_line = fig_competicao_plotly(df, UF, MET)
            st.plotly_chart(fig_line, use_container_width=True)

            # 2) Stacked 100% (market share) — colado abaixo
            fig_ms = fig_marketshare_100_plotly(df, UF, MET)
            st.plotly_chart(fig_ms, use_container_width=True)

            if UF == "RJ":
                fig_tipo = fig_rj_tipo_ebape_plotly(MET)
                st.plotly_chart(fig_tipo, use_container_width=True)




