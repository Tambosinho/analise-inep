# app.py — INEP Administração (Bacharelado)
# - Lê arquivos por ano de .ZIP
# - Filtros rígidos: Área=Negócios, Administração e Direito → Administração; Grau=Bacharelado; Modalidade∈{Presencial,EAD}
# - Visualizações + Auditoria + Market share + "Dados Gerais" (Instituições, Cursos, Vagas, Ingressantes, Matrículas, Concluintes)

import os
import json
import hashlib
import zipfile
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# PAGE / THEME
# -----------------------------
st.set_page_config(page_title="INEP • Administração (Bacharelado)", layout="wide")

PALETTE = {
    "azuis": ["#002D4D", "#003A79", "#0C63AA", "#008BC9", "#73BFE8"],
    "cinzas": ["#88868B", "#AFAEB4", "#D7D9DD"],
    "verde_claro": "#00E1AC",
    "azul_neon": "#01FFFF",
}
COLOR_CAT = {"Público": PALETTE["azuis"][0], "Privada c/ fins": PALETTE["azuis"][2], "Privada s/ fins": PALETTE["azuis"][4]}
COLOR_MOD = {"Presencial": PALETTE["azuis"][1], "EAD": PALETTE["azuis"][3]}
TITLE_COLOR, SUBTITLE_COLOR, GRID_COLOR, AXIS_COLOR = PALETTE["azuis"][0], PALETTE["cinzas"][0], PALETTE["cinzas"][2], PALETTE["cinzas"][1]
FONT_FAMILY = "Arial"

def format_short(n):
    if n is None or pd.isna(n): return ""
    n = float(n)
    if abs(n) >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000: return f"{n/1_000:.1f}k"
    return f"{int(n):d}"

# -----------------------------
# CONSTANTS / PATHS
# -----------------------------
ANOS = list(range(2010, 2024))
UF_FOCO_DEFAULT = "RJ"
DATA_ROOT_DEFAULT = r"C:\Users\joaod\OneDrive\Documentos\PROGESTAO\alvaro\analise-inep\dados"

APP_ROOT = os.getcwd()
CACHE_DIR = os.path.join(APP_ROOT, "_cache")
EXTRACT_DIR = os.path.join(CACHE_DIR, "extracted")
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(EXTRACT_DIR, exist_ok=True)

AREA_GERAL_TARGET = {"NEGÓCIOS, ADMINISTRAÇÃO E DIREITO"}
AREA_ESPECIFICA_TARGET = {"ADMINISTRAÇÃO"}
GRAU_TARGET = {"BACHARELADO"}
MODALIDADE_TARGET = {"Presencial", "EAD"}

COLMAP = {
    "CURSOS": {
        "ano": ["NU_ANO_CENSO", "NU_ANO"],
        "co_ies": ["CO_IES"],
        "co_curso": ["CO_CURSO", "CO_CURSO_ATUAL"],
        "sg_uf": ["SG_UF"],
        "no_curso": ["NO_CURSO", "NM_CURSO"],
        "tp_grau": ["TP_GRAU_ACADEMICO", "DS_GRAU_ACADEMICO"],
        "tp_modalidade": ["TP_MODALIDADE_ENSINO", "DS_MODALIDADE_ENSINO"],
        "qt_ing": ["QT_ING", "QT_INGRESSO"],
        "qt_mat": ["QT_MAT", "QT_MATRICULA"],
        "qt_conc": ["QT_CONC", "QT_CONCLUINTE"],
        "qt_vagas": ["QT_VAGAS", "QT_VG_TOTAL", "QT_VG_OFERTADAS", "QT_VG", "QT_VAC", "QT_VAG_TOTAL"],
        "area_geral": ["NO_CINE_AREA_GERAL","NO_GRANDE_AREA","NO_CINE_AREA","NO_AREA_CURSO","NO_CINE_AREA_DG","NO_CINE_AREA_GERAL_CURSO"],
        "area_especifica": ["NO_CINE_AREA_ESPECIFICA","NO_CINE_AREA_DETALHADA","NO_CINE_AREA_ESPECIFICA_CURSO","NO_CINE_AREA_DETALHADA_CURSO"],
    },
    "IES": {"co_ies": ["CO_IES"], "cat_adm": ["TP_CATEGORIA_ADMINISTRATIVA", "DS_CATEGORIA_ADMINISTRATIVA"]},
}

# -----------------------------
# HELPERS
# -----------------------------
def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def resolve_colnames(available_cols, synonyms_list):
    if not synonyms_list: return None
    av = set(available_cols)
    for c in synonyms_list:
        if c in av: return c
    norm_map = {_norm(c): c for c in available_cols}
    for c in synonyms_list:
        if _norm(c) in norm_map: return norm_map[_norm(c)]
    for c in synonyms_list:
        nc = _norm(c)
        hits = [orig for orig in available_cols if nc in _norm(orig)]
        if hits: return hits[0]
    return None

def map_modalidade(x):
    if pd.isna(x): return None
    try: return {1: "Presencial", 2: "EAD"}[int(float(str(x)))]
    except:
        xs = _norm(x)
        if "PRESEN" in xs: return "Presencial"
        if "DIST" in xs or "EAD" in xs: return "EAD"
        return None

def map_grau(x):
    if pd.isna(x): return None
    try: return {1: "BACHARELADO", 2: "LICENCIATURA", 3: "TECNOLÓGICO"}[int(float(str(x)))]
    except: return _norm(x)

def map_cat_adm_det(x):
    if pd.isna(x): return None
    try:
        return {1: "Pública Federal", 2: "Pública Estadual", 3: "Pública Municipal", 4: "Privada com fins", 5: "Privada sem fins"}[int(float(str(x)))]
    except:
        xs = _norm(x)
        if "FEDER" in xs: return "Pública Federal"
        if "ESTADU" in xs: return "Pública Estadual"
        if "MUNIC" in xs: return "Pública Municipal"
        if "SEM FINS" in xs: return "Privada sem fins"
        if "COM FINS" in xs or "PRIVADA" in xs: return "Privada com fins"
        return None

def to_macro_cat(cat_detalhada):
    if pd.isna(cat_detalhada): return None
    if str(cat_detalhada).startswith("Pública"): return "Público"
    if cat_detalhada == "Privada com fins": return "Privada c/ fins"
    if cat_detalhada == "Privada sem fins": return "Privada s/ fins"
    return None

# -----------------------------
# ZIP HANDLING
# -----------------------------
def _zip_path(data_root: str, year: int) -> str:
    return os.path.join(data_root, f"microdados_censo_da_educacao_superior_{year}.zip")

def _find_member(z: zipfile.ZipFile, filename_suffix: str) -> str | None:
    suffix_norm = filename_suffix.replace("\\", "/").lower()
    for name in z.namelist():
        if name.lower().endswith(suffix_norm):
            return name
    fname = os.path.basename(filename_suffix).lower()
    for name in z.namelist():
        if name.lower().endswith("/" + fname) or name.lower() == fname:
            return name
    return None

def _extract_from_zip(zip_path: str, internal_relpath: str, tag: str) -> str:
    h = hashlib.sha1(f"{os.path.basename(zip_path)}::{internal_relpath}".encode()).hexdigest()[:16]
    out_path = os.path.join(EXTRACT_DIR, f"{tag}_{h}.csv")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    with zipfile.ZipFile(zip_path, "r") as z:
        member = _find_member(z, internal_relpath) or _find_member(z, os.path.basename(internal_relpath))
        if member is None: raise FileNotFoundError(f"No member '{internal_relpath}' inside {zip_path}")
        with z.open(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path

def _direct_csv_path(data_root: str, year: int, kind: str) -> str:
    base = os.path.join(data_root, f"microdados_censo_da_educacao_superior_{year}", f"microdados_censo_da_educacao_superior_{year}", "dados")
    if kind == "CURSOS": return os.path.join(base, f"MICRODADOS_CADASTRO_CURSOS_{year}.CSV")
    return os.path.join(base, f"MICRODADOS_ED_SUP_IES_{year}.CSV")

def build_source_csv_path(data_root: str, year: int, kind: str) -> str:
    zip_file = _zip_path(data_root, year)
    if os.path.isfile(zip_file):
        base_rel = f"microdados_censo_da_educacao_superior_{year}/microdados_censo_da_educacao_superior_{year}/dados"
        rel = f"{base_rel}/MICRODADOS_CADASTRO_CURSOS_{year}.CSV" if kind == "CURSOS" else f"{base_rel}/MICRODADOS_ED_SUP_IES_{year}.CSV"
        tag = f"{kind}_{year}"
        return _extract_from_zip(zip_file, rel, tag)
    direct = _direct_csv_path(data_root, year, kind)
    if os.path.isfile(direct): return direct
    raise FileNotFoundError(f"CSV not found: year={year}, kind={kind}. Expected zip at {zip_file} or CSV at {direct}")

# -----------------------------
# DATA LOADERS (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_ies_map(year: int, data_root: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"ies_map_{year}.parquet")
    if os.path.exists(cache): return pd.read_parquet(cache)
    path = build_source_csv_path(data_root, year, "IES")
    head = pd.read_csv(path, sep=";", encoding="ISO-8859-1", nrows=0)
    cols = list(head.columns)
    co_ies = resolve_colnames(cols, COLMAP["IES"]["co_ies"])
    cat_adm = resolve_colnames(cols, COLMAP["IES"]["cat_adm"])
    if not co_ies or not cat_adm: raise RuntimeError(f"[IES {year}] Colunas essenciais ausentes.")
    df = pd.read_csv(path, sep=";", encoding="ISO-8859-1", usecols=[co_ies, cat_adm], dtype={co_ies:"int64"}, low_memory=True)
    df.rename(columns={co_ies:"CO_IES", cat_adm:"CAT_RAW"}, inplace=True)
    df["CAT_DETALHADA"] = df["CAT_RAW"].map(map_cat_adm_det)
    df["CAT_MACRO"] = df["CAT_DETALHADA"].map(to_macro_cat)
    df = df.dropna(subset=["CAT_MACRO"]).drop_duplicates(subset=["CO_IES"])[["CO_IES","CAT_MACRO"]]
    df["CAT_MACRO"] = df["CAT_MACRO"].astype("category")
    df.to_parquet(cache, index=False)
    return df

def _audit_path(year:int)->str: return os.path.join(CACHE_DIR, f"audit_{year}.json")
def _stats_path(year:int)->str: return os.path.join(CACHE_DIR, f"stats_{year}_adm.parquet")

@st.cache_data(show_spinner=True)
def process_year_to_agg(year: int, data_root: str, uf_foco: str = UF_FOCO_DEFAULT, chunksize: int = 200_000, rebuild_stats: bool=False) -> pd.DataFrame:
    """Processa 1 ano. Retorna o agregado (para os gráficos) e garante a criação do stats parquet."""
    cache_agg = os.path.join(CACHE_DIR, f"agg_{year}_adm.parquet")
    cache_stats = _stats_path(year)

    # Se já temos agg e (stats existe ou não exigimos rebuild), devolve o agg direto
    if os.path.exists(cache_agg) and (os.path.exists(cache_stats) or not rebuild_stats):
        return pd.read_parquet(cache_agg)

    path = build_source_csv_path(data_root, year, "CURSOS")
    ies_map = load_ies_map(year, data_root)

    head = pd.read_csv(path, sep=";", encoding="ISO-8859-1", nrows=0)
    cols = list(head.columns)
    res = lambda key: resolve_colnames(cols, COLMAP["CURSOS"][key])

    col_ano   = res("ano"); col_ies = res("co_ies"); col_curso_id = res("co_curso")
    col_uf    = res("sg_uf"); col_curso = res("no_curso")
    col_grau  = res("tp_grau"); col_mod = res("tp_modalidade")
    col_ing   = res("qt_ing"); col_mat = res("qt_mat"); col_con = res("qt_conc"); col_vagas = res("qt_vagas")
    col_area_g = res("area_geral"); col_area_e = res("area_especifica")

    needed = [col_ano, col_ies, col_uf, col_curso, col_grau, col_mod, col_ing, col_mat, col_con]
    if col_curso_id: needed.append(col_curso_id)
    if col_vagas: needed.append(col_vagas)
    if col_area_g: needed.append(col_area_g)
    if col_area_e: needed.append(col_area_e)
    if any(c is None for c in [col_ano, col_ies, col_uf, col_curso, col_grau, col_mod, col_ing, col_mat, col_con]):
        raise RuntimeError(f"[CURSOS {year}] Falta coluna essencial.")

    dtypes = {col_ies:"int64", col_ing:"Int32", col_mat:"Int32", col_con:"Int32"}
    if col_curso_id: dtypes[col_curso_id] = "Int64"
    if col_vagas: dtypes[col_vagas] = "Int64"

    # --- AUDIT counters
    audit = {
        "year": year, "rows_total": 0, "rows_area_geral": 0, "rows_area_especifica_or_nome": 0,
        "rows_grau_bacharelado": 0, "rows_modalidade_allowed": 0, "rows_after_merge_cat": 0,
        "modalidade_counts": {"Presencial": 0, "EAD": 0},
        "area_geral_seen": set(), "area_espec_seen": set(),
        "curso_id_col": col_curso_id, "vagas_col": col_vagas,
    }

    # --- acumuladores para "Dados Gerais"
    br_ies, rj_ies = set(), set()
    br_cursos, rj_cursos = set(), set()
    br_vagas = 0; rj_vagas = 0
    br_ing = 0; br_mat = 0; br_con = 0
    rj_ing = 0; rj_mat = 0; rj_con = 0

    parts = []
    for chunk in pd.read_csv(path, sep=";", encoding="ISO-8859-1", usecols=needed, dtype=dtypes, chunksize=chunksize, low_memory=True):
        audit["rows_total"] += len(chunk)
        chunk.rename(columns={
            col_ano:"ANO", col_ies:"CO_IES", col_uf:"SG_UF", col_curso:"NO_CURSO",
            col_grau:"TP_GRAU_ACADEMICO", col_mod:"TP_MODALIDADE_ENSINO",
            col_ing:"QT_ING", col_mat:"QT_MAT", col_con:"QT_CONC",
            **({col_curso_id:"CO_CURSO"} if col_curso_id else {}),
            **({col_vagas:"QT_VAGAS"} if col_vagas else {}),
            **({col_area_g:"AREA_GERAL"} if col_area_g else {}),
            **({col_area_e:"AREA_ESPEC"} if col_area_e else {}),
        }, inplace=True)

        # Área Geral
        if "AREA_GERAL" in chunk:
            m = chunk["AREA_GERAL"].astype(str).map(_norm).isin({_norm(v) for v in AREA_GERAL_TARGET})
            audit["rows_area_geral"] += int(m.sum())
            audit["area_geral_seen"].update(pd.unique(chunk.loc[m,"AREA_GERAL"].dropna().map(str)))
            chunk = chunk[m]

        # Área Específica (= Administração) ou fallback pelo nome
        if "AREA_ESPEC" in chunk:
            m = chunk["AREA_ESPEC"].astype(str).map(_norm).isin({_norm(v) for v in AREA_ESPECIFICA_TARGET})
            audit["rows_area_especifica_or_nome"] += int(m.sum())
            audit["area_espec_seen"].update(pd.unique(chunk.loc[m,"AREA_ESPEC"].dropna().map(str)))
            chunk = chunk[m]
        else:
            m = chunk["NO_CURSO"].astype(str).map(_norm) == _norm("ADMINISTRAÇÃO")
            audit["rows_area_especifica_or_nome"] += int(m.sum()); chunk = chunk[m]

        # Grau
        chunk["GRAU_PAD"] = chunk["TP_GRAU_ACADEMICO"].map(map_grau)
        m_grau = chunk["GRAU_PAD"].isin(GRAU_TARGET)
        audit["rows_grau_bacharelado"] += int(m_grau.sum()); chunk = chunk[m_grau]

        # Modalidade
        chunk["MODALIDADE"] = chunk["TP_MODALIDADE_ENSINO"].map(map_modalidade)
        m_mod = chunk["MODALIDADE"].isin(MODALIDADE_TARGET)
        audit["rows_modalidade_allowed"] += int(m_mod.sum()); chunk = chunk[m_mod]
        vc = chunk["MODALIDADE"].value_counts()
        for k in ["Presencial","EAD"]: audit["modalidade_counts"][k] += int(vc.get(k,0))

        # Categoria administrativa (via IES)
        chunk = chunk.merge(ies_map, on="CO_IES", how="left").dropna(subset=["CAT_MACRO","MODALIDADE"])
        audit["rows_after_merge_cat"] += len(chunk)

        # --- Atualiza acumuladores "Dados Gerais"
        br_ies.update(chunk["CO_IES"].dropna().astype(str).unique())
        if "CO_CURSO" in chunk: br_cursos.update(chunk["CO_CURSO"].dropna().astype(str).unique())
        elif "NO_CURSO" in chunk: br_cursos.update(chunk["NO_CURSO"].dropna().astype(str).unique())
        if "QT_VAGAS" in chunk: br_vagas += int(pd.to_numeric(chunk["QT_VAGAS"], errors="coerce").fillna(0).sum())
        br_ing += int(pd.to_numeric(chunk["QT_ING"], errors="coerce").fillna(0).sum())
        br_mat += int(pd.to_numeric(chunk["QT_MAT"], errors="coerce").fillna(0).sum())
        br_con += int(pd.to_numeric(chunk["QT_CONC"], errors="coerce").fillna(0).sum())

        rj = chunk[chunk["SG_UF"] == uf_foco]
        rj_ies.update(rj["CO_IES"].dropna().astype(str).unique())
        if "CO_CURSO" in rj: rj_cursos.update(rj["CO_CURSO"].dropna().astype(str).unique())
        elif "NO_CURSO" in rj: rj_cursos.update(rj["NO_CURSO"].dropna().astype(str).unique())
        if "QT_VAGAS" in rj: rj_vagas += int(pd.to_numeric(rj["QT_VAGAS"], errors="coerce").fillna(0).sum())
        rj_ing += int(pd.to_numeric(rj["QT_ING"], errors="coerce").fillna(0).sum())
        rj_mat += int(pd.to_numeric(rj["QT_MAT"], errors="coerce").fillna(0).sum())
        rj_con += int(pd.to_numeric(rj["QT_CONC"], errors="coerce").fillna(0).sum())

        # --- Agregados para gráficos (categoria x modalidade)
        for c in ["SG_UF","MODALIDADE","CAT_MACRO"]: chunk[c] = chunk[c].astype("category")
        br_grp = (chunk.groupby(["ANO","CAT_MACRO","MODALIDADE"], observed=True, as_index=False)[["QT_ING","QT_MAT","QT_CONC"]].sum(min_count=1))
        br_grp["ESCOPO"] = "BR"
        rj_grp = (rj.groupby(["ANO","CAT_MACRO","MODALIDADE"], observed=True, as_index=False)[["QT_ING","QT_MAT","QT_CONC"]].sum(min_count=1))
        rj_grp["ESCOPO"] = uf_foco
        parts.append(pd.concat([br_grp, rj_grp], ignore_index=True))

    # --- Salva agregados (gráficos)
    agg = pd.concat(parts, ignore_index=True)
    agg["ANO"] = int(year)
    for c in ["CAT_MACRO","MODALIDADE","ESCOPO"]: agg[c] = agg[c].astype("category")
    agg.sort_values(["ESCOPO","CAT_MACRO","MODALIDADE"], inplace=True)
    agg.to_parquet(cache_agg, index=False)

    # --- Salva "Dados Gerais"
    stats = pd.DataFrame([
        {"ANO":year,"ESCOPO":"BR","INSTITUICOES":len(br_ies),"CURSOS":len(br_cursos),"VAGAS":br_vagas,"QT_ING":br_ing,"QT_MAT":br_mat,"QT_CONC":br_con},
        {"ANO":year,"ESCOPO":uf_foco,"INSTITUICOES":len(rj_ies),"CURSOS":len(rj_cursos),"VAGAS":rj_vagas,"QT_ING":rj_ing,"QT_MAT":rj_mat,"QT_CONC":rj_con},
    ])
    stats.to_parquet(cache_stats, index=False)

    # --- Auditoria
    audit["area_geral_seen"] = sorted(list(audit["area_geral_seen"]))
    audit["area_espec_seen"] = sorted(list(audit["area_espec_seen"]))
    with open(_audit_path(year), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    return agg

@st.cache_data(show_spinner=True)
def build_master_agg(anos, data_root: str, uf_foco=UF_FOCO_DEFAULT):
    cache = os.path.join(CACHE_DIR, f"master_agg_{anos[0]}_{anos[-1]}_adm_{uf_foco}.parquet")
    if os.path.exists(cache): return pd.read_parquet(cache)
    out = []
    progress = st.progress(0.0, text="Processando anos…")
    for i, y in enumerate(anos, start=1):
        try: out.append(process_year_to_agg(y, data_root, uf_foco))
        except Exception as e: st.warning(f"[{y}] {e}")
        progress.progress(i/len(anos))
    progress.empty()
    if not out: return pd.DataFrame(columns=["ANO","ESCOPO","CAT_MACRO","MODALIDADE","QT_ING","QT_MAT","QT_CONC"])
    master = pd.concat(out, ignore_index=True)
    for c in ["QT_ING","QT_MAT","QT_CONC"]: master[c] = master[c].astype("Int64")
    for c in ["ESCOPO","CAT_MACRO","MODALIDADE"]: master[c] = master[c].astype("category")
    master.to_parquet(cache, index=False)
    return master

@st.cache_data(show_spinner=True)
def build_master_stats(anos, data_root: str, uf_foco=UF_FOCO_DEFAULT):
    rows = []
    for y in anos:
        p = _stats_path(y)
        if not os.path.exists(p):
            # força reprocessar estatísticas se não existirem
            process_year_to_agg(y, data_root, uf_foco, rebuild_stats=True)
        if os.path.exists(p):
            rows.append(pd.read_parquet(p))
    if not rows:
        return pd.DataFrame(columns=["ANO","ESCOPO","INSTITUICOES","CURSOS","VAGAS","QT_ING","QT_MAT","QT_CONC"])
    stats = pd.concat(rows, ignore_index=True)
    for c in ["INSTITUICOES","CURSOS","VAGAS","QT_ING","QT_MAT","QT_CONC"]:
        stats[c] = stats[c].astype("Int64")
    return stats

def load_audit_for_years(years):
    audits = []
    for y in years:
        p = _audit_path(y)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                audits.append(json.load(f))
    return audits

# -----------------------------
# SLICES / CHARTS
# -----------------------------
def slice_categoria(master: pd.DataFrame, escopo="BR"):
    return (master[master["ESCOPO"] == escopo]
            .groupby(["ANO", "CAT_MACRO"], observed=True, as_index=False)[["QT_ING", "QT_MAT", "QT_CONC"]]
            .sum(min_count=1))

def slice_modalidade(master: pd.DataFrame, escopo="BR"):
    return (master[master["ESCOPO"] == escopo]
            .groupby(["ANO", "MODALIDADE"], observed=True, as_index=False)[["QT_ING", "QT_MAT", "QT_CONC"]]
            .sum(min_count=1))

def compute_growth_table(df_long, group_col, value_col, yr_min, yr_max):
    base = (df_long[df_long["ANO"] == yr_min].set_index(group_col)[value_col].rename("base")).to_frame()
    end = (df_long[df_long["ANO"] == yr_max].set_index(group_col)[value_col].rename("end")).to_frame()
    tb = base.join(end, how="outer").fillna(0)
    years = max(1, yr_max - yr_min)
    tb["Δ absoluto"] = (tb["end"] - tb["base"]).astype(float)
    tb["CAGR"] = np.where((tb["base"] > 0) & (years >= 1),
                          (tb["end"] / tb["base"]).replace([np.inf, -np.inf], np.nan) ** (1/years) - 1,
                          np.nan)
    return tb.reset_index().sort_values("Δ absoluto", ascending=False)

def line_with_labels(df, x, y, series, color_map, title, subtitle, yaxis_title, show_labels=True):
    fig = go.Figure()
    for key, sub in df.groupby(series):
        sub = sub.sort_values(x)
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y],
            mode="lines+markers" + ("+text" if show_labels else ""),
            name=str(key),
            text=[format_short(v) for v in sub[y]] if show_labels else None,
            textposition="top center", textfont=dict(family=FONT_FAMILY, size=12),
            line=dict(color=color_map.get(str(key), PALETTE["azuis"][2]), width=3),
            marker=dict(size=7), cliponaxis=False,
            hovertemplate=f"{series}: {key}<br>{x}: %{{x}}<br>{yaxis_title}: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"<b style='color:{TITLE_COLOR}'>{title}</b><br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>{subtitle}</span>",
                   font=dict(family=FONT_FAMILY, size=20), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR, title=yaxis_title, tickformat=",.0f")
    return fig

def stacked_modalidade(df_mod, metric_col, title, subtitle, pct=False):
    pivot = df_mod.pivot(index="ANO", columns="MODALIDADE", values=metric_col).fillna(0).sort_index()
    if pct:
        totals = pivot.sum(axis=1).replace(0, np.nan)
        ydata = (pivot.div(totals, axis=0) * 100).fillna(0)
        yfmt = "%{y:.1f}%"
    else:
        ydata = pivot
        yfmt = "%{y:,.0f}"

    fig = go.Figure()
    for mod in ["Presencial", "EAD"]:
        if mod in ydata.columns:
            fig.add_trace(go.Bar(
                x=ydata.index, y=ydata[mod], name=mod,
                marker_color=COLOR_MOD.get(mod, PALETTE["azuis"][2]),
                text=[format_short(v) if not pct else f"{v:.0f}%" for v in ydata[mod]],
                textposition="inside" if pct else "auto",
                hovertemplate=f"Modalidade: {mod}<br>Ano: %{{x}}<br>Valor: {yfmt}<extra></extra>",
            ))
    fig.update_layout(
        barmode="stack",
        title=dict(text=f"<b style='color:{TITLE_COLOR}'>{title}</b><br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>{subtitle}</span>",
                   font=dict(family=FONT_FAMILY, size=20), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR,
                     title=("Participação (%)" if pct else "Total"), tickformat=(",.0f" if not pct else None), rangemode="tozero")
    return fig

# -----------------------------
# DEMO (opcional)
# -----------------------------
DEMO = os.getenv("DEMO", "0") == "1"
if DEMO:
    anos = list(range(2010, 2024))
    rows = []
    for escopo in ["BR", "RJ"]:
        for cat in ["Público", "Privada c/ fins", "Privada s/ fins"]:
            for mod in ["Presencial", "EAD"]:
                rng = np.random.default_rng(42 + hash((escopo, cat, mod)) % 10000)
                base = 18000 if escopo == "BR" else 1800
                drift = 0.02 if mod == "EAD" else 0.005
                vals = [base]
                for _ in anos[1:]:
                    vals.append(max(0, vals[-1] * (1 + drift + rng.normal(0, 0.03))))
                mats = [int(round(v)) for v in vals]
                conc = [int(round(v * 0.18)) for v in mats]
                ing  = [int(round(v * 0.22)) for v in mats]
                for ano, m, c, i in zip(anos, mats, conc, ing):
                    rows.append({"ANO": ano, "ESCOPO": escopo, "CAT_MACRO": cat, "MODALIDADE": mod, "QT_MAT": m, "QT_CONC": c, "QT_ING": i})
    DEMO_MASTER = pd.DataFrame(rows)
    # stats sintético (sem instituições/curso/vagas)
    DEMO_STATS = (DEMO_MASTER.groupby(["ANO","ESCOPO"], as_index=False)[["QT_ING","QT_MAT","QT_CONC"]].sum())
    DEMO_STATS["INSTITUICOES"] = pd.NA; DEMO_STATS["CURSOS"] = pd.NA; DEMO_STATS["VAGAS"] = pd.NA

# -----------------------------
# UI
# -----------------------------
st.markdown(
    f"<h2 style='color:{TITLE_COLOR};font-family:{FONT_FAMILY};font-weight:700;margin-bottom:0'>Administração (Bacharelado) — INEP 2010–2023</h2>"
    f"<p style='color:{SUBTITLE_COLOR};font-family:{FONT_FAMILY};margin-top:2px'>Comparativos: Público vs Privado e Presencial vs EAD • Brasil e RJ • Métricas: Ingressantes, Matrículas, Concluintes</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.subheader("Parâmetros")
    data_root = st.text_input("Pasta raiz dos microdados (contendo os .zip)", value=DATA_ROOT_DEFAULT)
    uf_foco = st.selectbox("Escopo secundário (UF)", options=[UF_FOCO_DEFAULT], index=0)
    anos_sel = st.slider("Período", min_value=min(ANOS), max_value=max(ANOS), value=(min(ANOS), max(ANOS)), step=1)
    metrica = st.radio("Métrica", ["Matrículas", "Concluintes", "Ingressantes"], index=0, horizontal=True)
    view = st.radio("Visualização", ["Por categoria administrativa", "Por modalidade"], index=0)
    ver_market_share = st.checkbox("Mostrar Market Share por Modalidade (barras empilhadas)", value=True)
    empilhado_pct = st.checkbox("Market Share 100% (participação)", value=True)
    show_labels = st.checkbox("Mostrar rótulos nos pontos (linhas)", value=True)
    mostrar_tabelas = st.checkbox("Mostrar tabelas resumo", value=False)
    ver_dep = st.checkbox("Mostrar 'Parâmetros aplicados' e 'Dados Gerais'", value=True)

# Build master + stats
if DEMO:
    master = DEMO_MASTER.copy()
    stats_all = DEMO_STATS.copy()
else:
    if not os.path.isdir(data_root):
        st.error("Pasta raiz inválida. Ela deve conter zips como microdados_censo_da_educacao_superior_2023.zip"); st.stop()
    try:
        master = build_master_agg(ANOS, data_root=data_root, uf_foco=uf_foco)
        stats_all = build_master_stats(ANOS, data_root=data_root, uf_foco=uf_foco)
    except Exception as e:
        st.exception(e); st.stop()

if master.empty:
    st.warning("Nenhum dado agregado. Verifique o caminho e os arquivos .zip."); st.stop()

yr_min, yr_max = anos_sel
metric_map = {"Matrículas":"QT_MAT","Concluintes":"QT_CONC","Ingressantes":"QT_ING"}
metric_col = metric_map[metrica]
m = master[(master["ANO"]>=yr_min) & (master["ANO"]<=yr_max)].copy()
stats = stats_all[(stats_all["ANO"]>=yr_min) & (stats_all["ANO"]<=yr_max)].copy()

# -----------------------------
# DEPURAÇÃO + DADOS GERAIS
# -----------------------------
if ver_dep:
    with st.expander("🧪 Depuração — Parâmetros aplicados & Dados Gerais", expanded=True):
        st.markdown(
            f"""
            **Parâmetros aplicados**
            - **Área Geral**: Negócios, Administração e Direito  
            - **Área Específica**: Administração  
            - **Grau**: Bacharelado  
            - **Modalidades**: Presencial e EAD  
            - **Escopos**: Brasil e {uf_foco}  
            - **Período**: {yr_min}–{yr_max}  
            - **Fonte**: CADASTRO_CURSOS + IES (INEP). Leituras por **chunks** direto do `.zip`.
            """.strip()
        )
        tabDG_br, tabDG_rj = st.tabs(["🇧🇷 Dados Gerais — Brasil", f"🏷️ Dados Gerais — {uf_foco}"])
        def show_stats(scope):
            df = (stats[stats["ESCOPO"]==scope][["ANO","INSTITUICOES","CURSOS","VAGAS","QT_ING","QT_MAT","QT_CONC"]]
                  .sort_values("ANO"))
            df = df.rename(columns={"QT_ING":"Ingressantes","QT_MAT":"Matrículas","QT_CONC":"Concluintes",
                                    "INSTITUICOES":"Instituições","CURSOS":"Cursos"})
            sty = (df.style.format({"Instituições":"{:,.0f}","Cursos":"{:,.0f}","VAGAS":"{:,.0f}",
                                    "Ingressantes":"{:,.0f}","Matrículas":"{:,.0f}","Concluintes":"{:,.0f}"})
                        .hide(axis="index"))
            st.dataframe(sty, use_container_width=True)
        with tabDG_br: show_stats("BR")
        with tabDG_rj: show_stats(UF_FOCO_DEFAULT)

# -----------------------------
# AUDITORIA (continua disponível)
# -----------------------------
with st.expander("🔎 Auditoria dos filtros (confirma Área→Administração, Grau=Bacharelado, Modalidade=Presencial/EAD)", expanded=False):
    audits = []
    for y in range(yr_min, yr_max+1):
        p = _audit_path(y)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f: audits.append(json.load(f))
    if not audits:
        st.info("A auditoria é criada ao processar cada ano. Reprocese os anos no período para vê-la aqui.")
    else:
        tot = {"rows_total":0,"rows_area_geral":0,"rows_area_especifica_or_nome":0,"rows_grau_bacharelado":0,"rows_modalidade_allowed":0,"rows_after_merge_cat":0,"Presencial":0,"EAD":0}
        area_geral_seen, area_espec_seen = set(), set()
        for a in audits:
            for k in ["rows_total","rows_area_geral","rows_area_especifica_or_nome","rows_grau_bacharelado","rows_modalidade_allowed","rows_after_merge_cat"]:
                tot[k] += a.get(k,0)
            modc = a.get("modalidade_counts",{})
            tot["Presencial"] += modc.get("Presencial",0); tot["EAD"] += modc.get("EAD",0)
            area_geral_seen.update(a.get("area_geral_seen",[])); area_espec_seen.update(a.get("area_espec_seen",[]))
        st.dataframe(pd.DataFrame({
            "Etapa":[
                "Linhas totais lidas","Área Geral = Negócios, Adm. e Direito",
                "Área Específica = Administração (ou Nome do curso)","Grau = Bacharelado",
                "Modalidade ∈ {Presencial, EAD}","Após merge com IES (categoria válida)"
            ],
            "Linhas":[tot["rows_total"], tot["rows_area_geral"], tot["rows_area_especifica_or_nome"], tot["rows_grau_bacharelado"], tot["rows_modalidade_allowed"], tot["rows_after_merge_cat"]],
        }), use_container_width=True, hide_index=True)
        st.write("**Modalidades (linhas) após filtros:**", f"Presencial={tot['Presencial']:,} | EAD={tot['EAD']:,}")
        st.write("**Áreas Gerais vistas:**", ", ".join(sorted(area_geral_seen)) or "—")
        st.write("**Áreas Específicas vistas:**", ", ".join(sorted(area_espec_seen)) or "—")

# -----------------------------
# TABS BR / RJ + VISUAIS
# -----------------------------
def render_scope(scope_label, escopo_df):
    col1, col2 = st.columns([2,1], gap="large")
    if view == "Por categoria administrativa":
        df_cat = slice_categoria(escopo_df, escopo=scope_label)
        fig = line_with_labels(df_cat, x="ANO", y=metric_col, series="CAT_MACRO", color_map=COLOR_CAT,
                               title=f"{metrica} por Categoria Administrativa",
                               subtitle=f"Administração (Bacharelado) • {('Brasil' if scope_label=='BR' else scope_label)} • {yr_min}–{yr_max}",
                               yaxis_title=metrica, show_labels=show_labels)
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_cat.rename(columns={"CAT_MACRO":"GRUPO"}), "GRUPO", metric_col, yr_min, yr_max)
    else:
        df_mod = slice_modalidade(escopo_df, escopo=scope_label)
        fig = line_with_labels(df_mod, x="ANO", y=metric_col, series="MODALIDADE", color_map=COLOR_MOD,
                               title=f"{metrica} por Modalidade",
                               subtitle=f"Administração (Bacharelado) • {('Brasil' if scope_label=='BR' else scope_label)} • {yr_min}–{yr_max}",
                               yaxis_title=metrica, show_labels=show_labels)
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_mod.rename(columns={"MODALIDADE":"GRUPO"}), "GRUPO", metric_col, yr_min, yr_max)

    if not tb.empty:
        tb["Δ absoluto"] = tb["Δ absoluto"].round(0).astype(int); tb["CAGR (%)"] = (tb["CAGR"]*100).round(2)
        tb_display = tb[["GRUPO","base","end","Δ absoluto","CAGR (%)"]]
        if mostrar_tabelas:
            col2.subheader("Crescimento no período"); col2.dataframe(tb_display, use_container_width=True, hide_index=True)
        top_gain, bottom = tb_display.iloc[0], tb_display.iloc[-1]
        col2.metric("Maior ganho (Δ)", f"{top_gain['GRUPO']}: {format_short(top_gain['Δ absoluto'])}")
        col2.metric("Maior perda (Δ)", f"{bottom['GRUPO']}: {format_short(bottom['Δ absoluto'])}")
        best_cagr = tb.sort_values("CAGR", ascending=False).iloc[0]
        col2.metric("Maior CAGR", f"{best_cagr['GRUPO']}: {(best_cagr['CAGR']*100):.2f}%")

    # Market share
    st.markdown("**Market share por Modalidade**")
    df_mod2 = slice_modalidade(escopo_df, escopo=scope_label)
    st.plotly_chart(
        stacked_modalidade(df_mod2, metric_col, title=f"{metrica} por Modalidade (Empilhado)", subtitle=f"{'Brasil' if scope_label=='BR' else scope_label} • {yr_min}–{yr_max}", pct=False),
        use_container_width=True,
    )
    if empilhado_pct:
        st.plotly_chart(
            stacked_modalidade(df_mod2, metric_col, title=f"Participação (%) por Modalidade (100%)", subtitle=f"{'Brasil' if scope_label=='BR' else scope_label} • {yr_min}–{yr_max}", pct=True),
            use_container_width=True,
        )

tab_br, tab_rj = st.tabs(["🇧🇷 Brasil", f"🏷️ {UF_FOCO_DEFAULT}"])
with tab_br: render_scope("BR", m)
with tab_rj: render_scope(UF_FOCO_DEFAULT, m)

# -----------------------------
# DOWNLOADS
# -----------------------------
st.divider()
parquet_path = os.path.join(CACHE_DIR, f"master_agg_{ANOS[0]}_{ANOS[-1]}_adm_{UF_FOCO_DEFAULT}.parquet")
colA, colB = st.columns(2)
if os.path.exists(parquet_path):
    with open(parquet_path, "rb") as f:
        colA.download_button("⬇️ Baixar agregado (Parquet)", data=f.read(), file_name=f"master_agg_adm_{ANOS[0]}_{ANOS[-1]}.parquet", mime="application/octet-stream")
else:
    colA.info("Arquivo agregado ainda não disponível.")
csv_buf = (m[["ANO","ESCOPO","CAT_MACRO","MODALIDADE","QT_ING","QT_MAT","QT_CONC"]].to_csv(index=False).encode("utf-8"))
colB.download_button("⬇️ Baixar dados filtrados (CSV)", data=csv_buf, file_name=f"adm_filtrado_{yr_min}_{yr_max}.csv", mime="text/csv")
