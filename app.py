# app.py — FIXED & HARDENED
# Run: streamlit run app.py

# ---- DEMO MODE: renders even without local files (for Streamlit Cloud) ----
import os
import pandas as pd
import numpy as np

DEMO = os.getenv("DEMO", "1") == "1"  # default ON in cloud

if DEMO:
    # Synthetic dataset: same schema as our master aggregate
    anos = list(range(2010, 2024))
    def mk_series(name, base, drift):
        rng = np.random.default_rng(42 + hash(name) % 1000)
        vals = [base]
        for _ in anos[1:]:
            vals.append(max(0, vals[-1] * (1 + drift + rng.normal(0, 0.03))))
        return [round(v) for v in vals]

    rows = []
    for escopo in ["BR", "RJ"]:
        for cat in ["Público", "Privada c/ fins", "Privada s/ fins"]:
            for mod in ["Presencial", "EAD"]:
                mats = mk_series(f"{escopo}-{cat}-{mod}-M", 20000 if escopo=="BR" else 2000, 0.02 if mod=="EAD" else 0.005)
                conc = [round(v * 0.18) for v in mats]
                ing  = [round(v * 0.22) for v in mats]
                for ano, qt_m, qt_c, qt_i in zip(anos, mats, conc, ing):
                    rows.append({"ANO": ano, "ESCOPO": escopo, "CAT_MACRO": cat, "MODALIDADE": mod,
                                 "QT_MAT": qt_m, "QT_CONC": qt_c, "QT_ING": qt_i})
    master = pd.DataFrame(rows)
    # Monkey-patch the functions the app expects:
    def build_master_agg(*args, **kwargs): return master.copy()
    def slice_categoria(df, escopo="BR"):
        return (df[df["ESCOPO"]==escopo]
                .groupby(["ANO","CAT_MACRO"], as_index=False)[["QT_ING","QT_MAT","QT_CONC"]].sum())
    def slice_modalidade(df, escopo="BR"):
        return (df[df["ESCOPO"]==escopo]
                .groupby(["ANO","MODALIDADE"], as_index=False)[["QT_ING","QT_MAT","QT_CONC"]].sum())
    # Also define ANOS and default UF so the rest of the file runs:
    ANOS = list(range(2010, 2024))
    UF_FOCO_DEFAULT = "RJ"
# ---- end DEMO MODE block ----
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

# -----------------------------
# CONSTANTS / PATHS
# -----------------------------
ANOS = list(range(2010, 2024))
UF_FOCO_DEFAULT = "RJ"

DATA_ROOT = r"C:\SEU\CAMINHO\ATE\AS\PASTAS"  # <-- adjust
CACHE_DIR = os.path.join(DATA_ROOT, "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

AREA_GERAL_TARGET = {"NEGÓCIOS, ADMINISTRAÇÃO E DIREITO"}
AREA_ESPECIFICA_TARGET = {"ADMINISTRAÇÃO"}
GRAU_TARGET = {"BACHARELADO"}
MODALIDADE_TARGET = {"Presencial", "EAD"}

COLMAP = {
    "CURSOS": {
        "ano": ["NU_ANO_CENSO", "NU_ANO"],
        "co_ies": ["CO_IES"],
        "sg_uf": ["SG_UF"],
        "no_curso": ["NO_CURSO", "NM_CURSO"],
        "tp_grau": ["TP_GRAU_ACADEMICO", "DS_GRAU_ACADEMICO"],
        "tp_modalidade": ["TP_MODALIDADE_ENSINO", "DS_MODALIDADE_ENSINO"],
        "qt_ing": ["QT_ING", "QT_INGRESSO"],
        "qt_mat": ["QT_MAT", "QT_MATRICULA"],
        "qt_conc": ["QT_CONC", "QT_CONCLUINTE"],
        "area_geral": ["NO_CINE_AREA_GERAL", "NO_GRANDE_AREA", "NO_CINE_AREA", "NO_AREA_CURSO", "NO_CINE_AREA_DG", "NO_CINE_AREA_GERAL_CURSO"],
        "area_especifica": ["NO_CINE_AREA_ESPECIFICA", "NO_CINE_AREA_DETALHADA", "NO_CINE_AREA_ESPECIFICA_CURSO", "NO_CINE_AREA_DETALHADA_CURSO"],
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

def build_path(year: int, kind: str) -> str:
    base = os.path.join(DATA_ROOT, f"microdados_censo_da_educacao_superior_{year}", f"microdados_censo_da_educacao_superior_{year}", "dados")
    if kind == "CURSOS":
        return os.path.join(base, f"MICRODADOS_CADASTRO_CURSOS_{year}.CSV")
    return os.path.join(base, f"MICRODADOS_ED_SUP_IES_{year}.CSV")

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

def format_short(n):
    if n is None or pd.isna(n): return ""
    n = float(n)
    if abs(n) >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000: return f"{n/1_000:.1f}k"
    return f"{int(n):d}"

# -----------------------------
# DATA LOADERS (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_ies_map(year: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"ies_map_{year}.parquet")
    if os.path.exists(cache): return pd.read_parquet(cache)
    path = build_path(year, "IES")
    probe = pd.read_csv(path, sep=";", encoding="ISO-8859-1", nrows=0)
    cols = list(probe.columns)
    co_ies = resolve_colnames(cols, COLMAP["IES"]["co_ies"])
    cat_adm = resolve_colnames(cols, COLMAP["IES"]["cat_adm"])
    if not co_ies or not cat_adm:
        raise RuntimeError(f"[IES {year}] required columns not found.")
    df = pd.read_csv(path, sep=";", encoding="ISO-8859-1", usecols=[co_ies, cat_adm], dtype={co_ies: "int64"}, low_memory=True)
    df.rename(columns={co_ies: "CO_IES", cat_adm: "CAT_RAW"}, inplace=True)
    df["CAT_DETALHADA"] = df["CAT_RAW"].map(map_cat_adm_det)
    df["CAT_MACRO"] = df["CAT_DETALHADA"].map(to_macro_cat)
    df = df.dropna(subset=["CAT_MACRO"]).drop_duplicates(subset=["CO_IES"])[["CO_IES", "CAT_MACRO"]]
    df["CAT_MACRO"] = df["CAT_MACRO"].astype("category")
    df.to_parquet(cache, index=False)
    return df

@st.cache_data(show_spinner=True)
def process_year_to_agg(year: int, uf_foco: str = UF_FOCO_DEFAULT, chunksize: int = 200_000) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"agg_{year}_adm.parquet")
    if os.path.exists(cache): return pd.read_parquet(cache)
    path = build_path(year, "CURSOS")
    ies_map = load_ies_map(year)
    probe = pd.read_csv(path, sep=";", encoding="ISO-8859-1", nrows=0)
    cols = list(probe.columns)
    res = lambda key: resolve_colnames(cols, COLMAP["CURSOS"][key])
    col_ano, col_ies, col_uf, col_curso, col_grau, col_mod = res("ano"), res("co_ies"), res("sg_uf"), res("no_curso"), res("tp_grau"), res("tp_modalidade")
    col_ing, col_mat, col_con = res("qt_ing"), res("qt_mat"), res("qt_conc")
    col_area_g, col_area_e = res("area_geral"), res("area_especifica")
    needed = [col_ano, col_ies, col_uf, col_curso, col_grau, col_mod, col_ing, col_mat, col_con]
    if col_area_g: needed.append(col_area_g)
    if col_area_e: needed.append(col_area_e)
    if any(c is None for c in [col_ano, col_ies, col_uf, col_curso, col_grau, col_mod, col_ing, col_mat, col_con]):
        raise RuntimeError(f"[CURSOS {year}] required columns not found.")
    dtypes = {col_ies: "int64", col_ing: "Int32", col_mat: "Int32", col_con: "Int32"}
    agg_parts = []
    for chunk in pd.read_csv(path, sep=";", encoding="ISO-8859-1", usecols=needed, dtype=dtypes, chunksize=chunksize, low_memory=True):
        chunk.rename(columns={
            col_ano: "ANO", col_ies: "CO_IES", col_uf: "SG_UF", col_curso: "NO_CURSO",
            col_grau: "TP_GRAU_ACADEMICO", col_mod: "TP_MODALIDADE_ENSINO",
            col_ing: "QT_ING", col_mat: "QT_MAT", col_con: "QT_CONC",
            **({col_area_g: "AREA_GERAL"} if col_area_g else {}),
            **({col_area_e: "AREA_ESPEC"} if col_area_e else {}),
        }, inplace=True)

        if "AREA_GERAL" in chunk:
            chunk = chunk[chunk["AREA_GERAL"].astype(str).map(_norm).isin({_norm(v) for v in AREA_GERAL_TARGET})]
        if "AREA_ESPEC" in chunk:
            chunk = chunk[chunk["AREA_ESPEC"].astype(str).map(_norm).isin({_norm(v) for v in AREA_ESPECIFICA_TARGET})]
        else:
            chunk = chunk[chunk["NO_CURSO"].astype(str).map(_norm) == _norm("ADMINISTRAÇÃO")]

        chunk["GRAU_PAD"] = chunk["TP_GRAU_ACADEMICO"].map(map_grau)
        chunk = chunk[chunk["GRAU_PAD"].isin(GRAU_TARGET)]

        chunk["MODALIDADE"] = chunk["TP_MODALIDADE_ENSINO"].map(map_modalidade)
        chunk = chunk[chunk["MODALIDADE"].isin(MODALIDADE_TARGET)]

        chunk = chunk.merge(ies_map, on="CO_IES", how="left").dropna(subset=["CAT_MACRO", "MODALIDADE"])
        for c in ["SG_UF", "MODALIDADE", "CAT_MACRO"]:
            chunk[c] = chunk[c].astype("category")

        br = (chunk.groupby(["ANO", "CAT_MACRO", "MODALIDADE"], observed=True, as_index=False)[["QT_ING", "QT_MAT", "QT_CONC"]].sum(min_count=1))
        br["ESCOPO"] = "BR"
        rj = chunk[chunk["SG_UF"] == uf_foco]
        rj = (rj.groupby(["ANO", "CAT_MACRO", "MODALIDADE"], observed=True, as_index=False)[["QT_ING", "QT_MAT", "QT_CONC"]].sum(min_count=1))
        rj["ESCOPO"] = uf_foco
        agg_parts.append(pd.concat([br, rj], ignore_index=True))

    agg = pd.concat(agg_parts, ignore_index=True)
    agg["ANO"] = int(year)
    for c in ["CAT_MACRO", "MODALIDADE", "ESCOPO"]:
        agg[c] = agg[c].astype("category")
    agg.sort_values(["ESCOPO", "CAT_MACRO", "MODALIDADE"], inplace=True)
    agg.to_parquet(cache, index=False)
    return agg

@st.cache_data(show_spinner=True)
def build_master_agg(anos=ANOS, uf_foco=UF_FOCO_DEFAULT):
    cache = os.path.join(CACHE_DIR, f"master_agg_{anos[0]}_{anos[-1]}_adm_{uf_foco}.parquet")
    if os.path.exists(cache): return pd.read_parquet(cache)
    out = []
    progress = st.progress(0.0, text="Processando anos…")
    for i, y in enumerate(anos, start=1):
        try:
            out.append(process_year_to_agg(y, uf_foco))
        except Exception as e:
            st.warning(f"[{y}] {e}")
        progress.progress(i / len(anos))
    progress.empty()
    if not out:
        return pd.DataFrame(columns=["ANO","ESCOPO","CAT_MACRO","MODALIDADE","QT_ING","QT_MAT","QT_CONC"])
    master = pd.concat(out, ignore_index=True)
    for c in ["QT_ING", "QT_MAT", "QT_CONC"]:
        master[c] = master[c].astype("Int64")
    for c in ["ESCOPO", "CAT_MACRO", "MODALIDADE"]:
        master[c] = master[c].astype("category")
    master.to_parquet(cache, index=False)
    return master

# -----------------------------
# SLICES / METRICS / CHARTS
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
    tb = tb.reset_index().sort_values("Δ absoluto", ascending=False)
    return tb

def line_with_labels(df, x, y, series, color_map, title, subtitle, yaxis_title):
    fig = go.Figure()
    for key, sub in df.groupby(series):
        sub = sub.sort_values(x)
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="lines+markers+text", name=str(key),
            text=[format_short(v) for v in sub[y]], textposition="top center",
            textfont=dict(family=FONT_FAMILY, size=12),
            line=dict(color=color_map.get(str(key), PALETTE["azuis"][2]), width=3),
            marker=dict(size=8),
            hovertemplate=f"{series}: {key}<br>{x}: %{{x}}<br>{yaxis_title}: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"<b style='color:{TITLE_COLOR}'>{title}</b><br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>{subtitle}</span>",
                   font=dict(family=FONT_FAMILY, size=20), x=0, xanchor="left", y=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR, title=yaxis_title)
    return fig

# -----------------------------
# UI
# -----------------------------
st.markdown(
    f"<h2 style='color:{TITLE_COLOR};font-family:{FONT_FAMILY};font-weight:700;margin-bottom:0'>Administração (Bacharelado) — INEP 2010–2023</h2>"
    f"<p style='color:{SUBTITLE_COLOR};font-family:{FONT_FAMILY};margin-top:2px'>Comparativos: Público vs Privado e Presencial vs EAD • Brasil e RJ • Métricas: Ingressantes, Matrículas, Concluintes</p>",
    unsafe_allow_html=True
)

with st.expander("Diagnostics", expanded=False):
    st.write("DATA_ROOT:", DATA_ROOT)
    st.write("CACHE_DIR exists:", os.path.isdir(CACHE_DIR))
    st.write("Years:", ANOS[0], "…", ANOS[-1])

with st.sidebar:
    st.subheader("Parâmetros")
    data_root_input = st.text_input("Pasta raiz dos microdados", value=DATA_ROOT)
    if data_root_input and data_root_input != DATA_ROOT:
        DATA_ROOT = data_root_input
        CACHE_DIR = os.path.join(DATA_ROOT, "_cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
    uf_foco = st.selectbox("Escopo secundário (UF)", options=[UF_FOCO_DEFAULT], index=0)
    anos_sel = st.slider("Período", min_value=min(ANOS), max_value=max(ANOS), value=(min(ANOS), max(ANOS)), step=1)
    metrica = st.radio("Métrica", ["Matrículas", "Concluintes", "Ingressantes"], index=0, horizontal=True)
    view = st.radio("Visualização", ["Por categoria administrativa", "Por modalidade"], index=0)
    mostrar_tabelas = st.checkbox("Mostrar tabelas resumo", value=False)

if not os.path.isdir(DATA_ROOT):
    st.error("Pasta raiz inválida. Ajuste na barra lateral.")
    st.stop()

# Build master (guarded)
try:
    master = build_master_agg(ANOS, uf_foco=uf_foco)
except Exception as e:
    st.error(f"Erro ao agregar dados: {e}")
    st.stop()

if master.empty:
    st.warning("Nenhum dado agregado. Verifique o caminho e os arquivos dos anos selecionados.")
    st.stop()

yr_min, yr_max = anos_sel
m = master[(master["ANO"] >= yr_min) & (master["ANO"] <= yr_max)].copy()
metric_map = {"Matrículas": "QT_MAT", "Concluintes": "QT_CONC", "Ingressantes": "QT_ING"}
metric_col = metric_map[metrica]

tab_br, tab_rj = st.tabs(["🇧🇷 Brasil", f"🏷️ {uf_foco}"])

# --------- BRASIL ---------
with tab_br:
    col1, col2 = st.columns([2,1], gap="large")
    if view == "Por categoria administrativa":
        df_cat = slice_categoria(m, escopo="BR")
        fig = line_with_labels(df_cat, x="ANO", y=metric_col, series="CAT_MACRO", color_map=COLOR_CAT,
                               title=f"{metrica} por Categoria Administrativa",
                               subtitle=f"Administração (Bacharelado) • Brasil • {yr_min}–{yr_max}",
                               yaxis_title=metrica)
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_cat.rename(columns={"CAT_MACRO":"GRUPO"}), "GRUPO", metric_col, yr_min, yr_max)
    else:
        df_mod = slice_modalidade(m, escopo="BR")
        fig = line_with_labels(df_mod, x="ANO", y=metric_col, series="MODALIDADE", color_map=COLOR_MOD,
                               title=f"{metrica} por Modalidade",
                               subtitle=f"Administração (Bacharelado) • Brasil • {yr_min}–{yr_max}",
                               yaxis_title=metrica)
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_mod.rename(columns={"MODALIDADE":"GRUPO"}), "GRUPO", metric_col, yr_min, yr_max)

    if not tb.empty:
        tb["Δ absoluto"] = tb["Δ absoluto"].round(0).astype(int)
        tb["CAGR (%)"] = (tb["CAGR"] * 100).round(2)
        tb_display = tb[["GRUPO","base","end","Δ absoluto","CAGR (%)"]]
        if mostrar_tabelas:
            col2.subheader("Crescimento no período")
            col2.dataframe(tb_display, use_container_width=True, hide_index=True)
        top_gain = tb_display.iloc[0]
        bottom = tb_display.iloc[-1]
        col2.metric("Maior ganho (Δ)", f"{top_gain['GRUPO']}: {format_short(top_gain['Δ absoluto'])}")
        col2.metric("Maior perda (Δ)", f"{bottom['GRUPO']}: {format_short(bottom['Δ absoluto'])}")
        best_cagr = tb.sort_values("CAGR", ascending=False).iloc[0]
        col2.metric("Maior CAGR", f"{best_cagr['GRUPO']}: {(best_cagr['CAGR']*100):.2f}%")

# --------- RJ ---------
with tab_rj:
    col1, col2 = st.columns([2,1], gap="large")
    if view == "Por categoria administrativa":
        df_cat = slice_categoria(m, escopo=uf_foco)
        fig = line_with_labels(df_cat, x="ANO", y=metric_col, series="CAT_MACRO", color_map=COLOR_CAT,
                               title=f"{metrica} por Categoria Administrativa",
                               subtitle=f"Administração (Bacharelado) • {uf_foco} • {yr_min}–{yr_max}",
                               yaxis_title=metrica)
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_cat.rename(columns={"CAT_MACRO":"GRUPO"}), "GRUPO", metric_col, yr_min, yr_max)
    else:
        df_mod = slice_modalidade(m, escopo=uf_foco)
        fig = line_with_labels(df_mod, x="ANO", y=metric_col, series="MODALIDADE", color_map=COLOR_MOD,
                               title=f"{metrica} por Modalidade",
                               subtitle=f"Administração (Bacharelado) • {uf_foco} • {yr_min}–{yr_max}",
                               yaxis_title=metrica)
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_mod.rename(columns={"MODALIDADE":"GRUPO"}), "GRUPO", metric_col, yr_min, yr_max)

    if not tb.empty:
        tb["Δ absoluto"] = tb["Δ absoluto"].round(0).astype(int)
        tb["CAGR (%)"] = (tb["CAGR"] * 100).round(2)
        tb_display = tb[["GRUPO","base","end","Δ absoluto","CAGR (%)"]]
        if mostrar_tabelas:
            col2.subheader("Crescimento no período")
            col2.dataframe(tb_display, use_container_width=True, hide_index=True)
        top_gain = tb_display.iloc[0]
        bottom = tb_display.iloc[-1]
        col2.metric("Maior ganho (Δ)", f"{top_gain['GRUPO']}: {format_short(top_gain['Δ absoluto'])}")
        col2.metric("Maior perda (Δ)", f"{bottom['GRUPO']}: {format_short(bottom['Δ absoluto'])}")
        best_cagr = tb.sort_values("CAGR", ascending=False).iloc[0]
        # ✅ fixed f-string here:
        col2.metric("Maior CAGR", f"{best_cagr['GRUPO']}: {(best_cagr['CAGR']*100):.2f}%")

# -----------------------------
# DOWNLOADS (only if file exists)
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
