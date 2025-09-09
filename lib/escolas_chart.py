# lib/escolas_charts.py
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st

# ========== CONFIG ==========
CSV_PATH = "planilha-escolas-rj-sp-mg-from-powerbi.csv"  # hardcoded

BLUES = {"navy":"#002D4D", "sky":"#73BFE8", "medium":"#0C63AA"}
GRAYS = {"mid":"#88868B", "light":"#D7D9DD", "white":"#FFFFFF"}

# não-azuis (manual + vibrantes, sem azuis)
NON_BLUE_HEX = [
    "#BDA132", "#B06F0B", "#7F1343", "#78487F", "#2B8671", "#668040",
    "#A02BAD", "#C680CE", "#782082", "#00E1AC", "#66EDCD", "#00A981",
]

# “força” visual
EBAPE_LW   = 3.0
EAESP_LW   = 2.6
OTHERS_LW  = 1.1
OTHERS_ALPHA = 0.55
OTHERS_GRAY_MIX = 0.40  # puxa demais para cinza → EBAPE/EAESP ganham destaque

# espaçamentos
SUBTITLE_PAD_EM = 1.0
LEGEND_GAP_EM   = 0.75
SUPTITLE_Y      = 0.995  # ↑ aumenta gap Título→Subtítulo

# Matplotlib base
plt.rcParams.update({
    "font.family": "Arial",
    "axes.edgecolor": GRAYS["light"],
    "axes.labelcolor": GRAYS["mid"],
    "text.color": GRAYS["mid"],
    "xtick.color": GRAYS["mid"],
    "ytick.color": GRAYS["mid"],
})

# ========== HELPERS ==========
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
    if miss: raise KeyError(f"Faltam colunas: {miss}")
    if pd.api.types.is_numeric_dtype(df["IES"]): df["IES"] = df["IES"].astype(str)
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

def _em_to_axes_height(ax, fig, em_pt: float) -> float:
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    axes_h_px = bbox.height
    em_px = (fig.dpi / 72.0) * em_pt
    return em_px / axes_h_px

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0,2,4))
def _rgb_to_hex(rgb): return "#{:02X}{:02X}{:02X}".format(*rgb)
def _mix_to(hex_color: str, target_hex: str, p: float) -> str:
    r1,g1,b1 = _hex_to_rgb(hex_color); r2,g2,b2 = _hex_to_rgb(target_hex)
    r = int(r1 + (r2 - r1) * p); g = int(g1 + (g2 - g1) * p); b = int(b1 + (b2 - b1) * p)
    return _rgb_to_hex((r,g,b))

def palette_distinct_others(n: int) -> list:
    base = NON_BLUE_HEX[:]
    if n <= len(base): return base[:n]
    i = 0
    while len(base) < n:
        base.append(_mix_to(NON_BLUE_HEX[i % len(NON_BLUE_HEX)], GRAYS["mid"], 0.25)); i += 1
    return base[:n]

# ========== DATA ==========
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df_raw = _read_csv_robusto(CSV_PATH)
    df = _mapear_colunas(df_raw)
    df = _coerce_numerics(df)
    df = (df.groupby(["UF","IES","Ano"], as_index=False)
            .agg({"Ingresso":"sum","Matriculas":"sum","Concluintes":"sum"}))
    df["IES_norm"] = df["IES"].map(lambda s: _strip_accents(s).lower())
    return df

# ========== PLOT ==========
def plot_metric_por_uf(
    df: pd.DataFrame, uf: str, metric: str,
    fig_width: float = 9.5, fig_height: float = 3.6, dpi: int = 180,
    value_fs: float = 6.5
) -> plt.Figure:
    title_map = {"Ingresso":"INGRESSO", "Matriculas":"MATRICULADOS", "Concluintes":"CONCLUINTES"}
    assert metric in title_map

    df_uf = df[df["UF"].str.fullmatch(uf, na=False)].copy()
    frames = [df_uf, df[df["IES_norm"].apply(is_ebape)]]
    if uf == "SP":
        frames.append(df[df["IES_norm"].apply(is_eaesp)])
    df_plot = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["UF","IES","Ano","Ingresso","Matriculas","Concluintes"]
    )

    anos = sorted(df_plot["Ano"].dropna().unique())
    if not anos: raise ValueError(f"Sem anos válidos para UF={uf}")

    # paleta
    unique_ies = sorted(df_plot["IES"].unique())
    ebapes = [i for i in unique_ies if is_ebape(_strip_accents(i).lower())]
    eaesps = [i for i in unique_ies if (uf == "SP" and is_eaesp(_strip_accents(i).lower()))] if uf == "SP" else []
    others = [i for i in unique_ies if i not in ebapes + eaesps]

    base_others = palette_distinct_others(len(others))
    cmap = {}
    for ies, col in zip(others, base_others):
        cmap[ies] = _mix_to(col, GRAYS["mid"], OTHERS_GRAY_MIX)  # enfraquece concorrentes
    for nm in ebapes: cmap[nm] = BLUES["sky"]
    for nm in eaesps: cmap[nm] = BLUES["navy"]

    order = others + eaesps + ebapes

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi, facecolor=GRAYS["white"])
    fig.suptitle(f"COMPETIÇÃO ENTRE ESCOLAS - {uf}", fontsize=16, fontweight="bold",
                 x=0.02, y=SUPTITLE_Y, ha="left", color=BLUES["navy"])

    subtitle_fs = 12
    ax.set_title(title_map[metric], fontsize=subtitle_fs, color=BLUES["medium"],
                 pad=subtitle_fs * SUBTITLE_PAD_EM, loc="left")

    ax.set_facecolor(GRAYS["white"])
    for s in ["top","right","left","bottom"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRAYS["light"], linewidth=0.8, alpha=0.9, zorder=0)
    ax.tick_params(labelsize=value_fs+2)
    ax.set_xlabel("ANO", labelpad=value_fs, color=GRAYS["mid"])
    ax.set_ylabel("")

    handles, seen = [], set()
    for ies in order:
        dat = df_plot[df_plot["IES"] == ies][["Ano", metric]].dropna().sort_values("Ano")
        if dat.empty: continue
        xs, ys = dat["Ano"].values, dat[metric].values
        name_norm = _strip_accents(ies).lower()
        is_EBAPE = is_ebape(name_norm)
        is_EAESP = (uf == "SP" and is_eaesp(name_norm))
        lw    = EBAPE_LW if is_EBAPE else (EAESP_LW if is_EAESP else OTHERS_LW)
        alpha = 1.0       if (is_EBAPE or is_EAESP) else OTHERS_ALPHA
        z     = 3.5       if is_EBAPE else (3.0 if is_EAESP else 2.0)
        color = cmap.get(ies, GRAYS["mid"])

        ax.plot(xs, ys, "-o", color=color, lw=lw, ms=3.0, alpha=alpha, zorder=z)

        dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.016
        label_color = color if (is_EBAPE or is_EAESP) else _mix_to(color, GRAYS["mid"], 0.15)
        label_alpha = 1.0 if (is_EBAPE or is_EAESP) else max(0.5, OTHERS_ALPHA - 0.05)
        for x, y in zip(xs, ys):
            ax.text(x, y + dy, _fmt_int(y), fontsize=value_fs, color=label_color,
                    alpha=label_alpha, ha="center", va="bottom", zorder=z-0.8)

        if ies not in seen:
            handles.append(Line2D([0],[0], color=color, lw=lw, alpha=alpha, label=ies))
            seen.add(ies)

    ax.set_xticks(anos)
    ax.set_xlim(min(anos)-0.2, max(anos)+0.2)

    legend_y = -_em_to_axes_height(ax, fig, value_fs * LEGEND_GAP_EM)
    ncol = min(10, max(2, int(np.ceil(len(handles) / 2))))
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(0.0, legend_y), bbox_transform=ax.transAxes,
              borderaxespad=0.0, frameon=False, fontsize=value_fs,
              ncol=ncol, columnspacing=1.0, handlelength=2.6)

    return fig
