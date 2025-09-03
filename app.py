# app.py — INEP Administração (Bacharelado) • pronto para deploy com senha
# Leitura direta de adm_bacharelado.csv (pré-filtrado)


def require_password() -> bool:
    """Simple one-password gate.
    - Reads the password from st.secrets['password'] or env APP_PASSWORD.
    - Shows a small login form until the user authenticates.
    - After success, the form disappears (using rerun).
    """
    # Already authed this session?
    if st.session_state.get("auth_ok", False):
        return True

    real = st.secrets.get("password") or os.getenv("APP_PASSWORD")

    # Enforce having a password configured (safer default)
    if not real:
        st.error("Password not configured. Set st.secrets['password'] or env APP_PASSWORD.")
        st.stop()

    # Render login form inside a placeholder so we can remove it
    gate = st.empty()
    with gate.form("login_form", clear_on_submit=True):
        st.markdown("### 🔒 Enter password")
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        if pwd == real:
            # mark session authed and hide the form
            st.session_state["auth_ok"] = True
            gate.empty()
            st.rerun()  # reload app without the login UI
        else:
            st.error("Invalid password.")
            # keep the form on screen this run
            return False

    # Not submitted yet → block the rest of the app
    st.stop()

# Call the gate before any other app content:
require_password()

# (Optional) Logout button for convenience
with st.sidebar:
    if st.session_state.get("auth_ok"):
        if st.button("Logout"):
            st.session_state.pop("auth_ok", None)
            st.rerun()


import os
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
TITLE_COLOR, SUBTITLE_COLOR, GRID_COLOR, AXIS_COLOR = (
    PALETTE["azuis"][0],
    PALETTE["cinzas"][0],
    PALETTE["cinzas"][2],
    PALETTE["cinzas"][1],
)
FONT_FAMILY = "Arial"

# cores finais dos grupos
COLOR_CAT = {
    "Público": PALETTE["azuis"][0],
    "Privada c/ fins": PALETTE["azuis"][2],
    "Privada s/ fins": PALETTE["verde_claro"],  # mais discriminação
    "Outro": PALETTE["cinzas"][1],              # cinza
}
COLOR_MOD = {
    "Presencial": PALETTE["azuis"][1],
    "EAD": PALETTE["azuis"][3],
    "Outro": PALETTE["cinzas"][2],
}


# -----------------------------
# SIMPLE AUTH (password)
# -----------------------------
def check_password() -> bool:
    """Ask for a password once per session. Source from st.secrets or APP_PASSWORD env."""
    if "password_ok" in st.session_state:
        return st.session_state.password_ok

    real = st.secrets.get("password") or os.getenv("APP_PASSWORD")
    if not real:
        # No password provided — allow access (useful in local dev)
        st.warning("⚠️ No password found in `st.secrets['password']` or env `APP_PASSWORD`. App is open.")
        st.session_state.password_ok = True
        return True

    with st.form("auth"):
        st.markdown("### 🔒 Enter password")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
    if ok:
        if pw == real:
            st.session_state.password_ok = True
            return True
        st.error("Wrong password.")
    return False


if not check_password():
    st.stop()


# -----------------------------
# HELPERS
# -----------------------------
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


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = "adm_bacharelado.csv"
    if not os.path.exists(csv_path):
        st.error("`adm_bacharelado.csv` não encontrado no diretório do app.")
        return pd.DataFrame()

    # Aceita ambas as convenções de nomes
    candidate_cols = {
        "ANO", "NU_ANO_CENSO",
        "QT_VG_TOTAL", "QT_ING", "QT_MAT", "QT_CONC",
        "TP_MODALIDADE_ENSINO",
        "CATEGORIA_ADMINISTRATIVA", "TP_CATEGORIA_ADMINISTRATIVA", "DS_CATEGORIA_ADMINISTRATIVA",
        "SG_UF", "NO_MUNICIPIO",
        "CO_IES", "NO_CURSO"
    }

    df = pd.read_csv(
        csv_path,
        sep=",",
        encoding="utf-8",
        low_memory=False,
        usecols=lambda c: c in candidate_cols
    )

    # padroniza nomes
    rename_map = {
        "NU_ANO_CENSO": "ANO",
        "QT_VG_TOTAL": "VAGAS",
    }
    df = df.rename(columns=rename_map)

    # numéricos principais
    for col in ["ANO", "VAGAS", "QT_ING", "QT_MAT", "QT_CONC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # coluna-fonte da categoria administrativa (NÃO sobrescrever se já existir)
    if "CATEGORIA_ADMINISTRATIVA" in df.columns:
        cat_raw = df["CATEGORIA_ADMINISTRATIVA"]
    elif "TP_CATEGORIA_ADMINISTRATIVA" in df.columns:
        cat_raw = df["TP_CATEGORIA_ADMINISTRATIVA"]
    elif "DS_CATEGORIA_ADMINISTRATIVA" in df.columns:
        cat_raw = df["DS_CATEGORIA_ADMINISTRATIVA"]
    else:
        cat_raw = pd.Series([np.nan] * len(df))
    df["CAT_RAW"] = cat_raw

    # mapeamento robusto (numérico ou textual)
    def map_cat(val):
        if pd.isna(val):
            return "Outro"
        # números: 1/2/3 público, 4 c/ fins, 5 s/ fins, 7 outro
        if isinstance(val, (int, np.integer, float, np.floating)):
            try:
                ival = int(val)
                if ival in (1, 2, 3):
                    return "Público"
                if ival == 4:
                    return "Privada c/ fins"
                if ival == 5:
                    return "Privada s/ fins"
                return "Outro"
            except Exception:
                return "Outro"
        # strings
        s = str(val).strip().lower()
        if s.isdigit():
            return map_cat(int(s))
        if "pública" in s:
            return "Público"
        if "privada" in s and "com" in s:
            return "Privada c/ fins"
        if "privada" in s and "sem" in s:
            return "Privada s/ fins"
        return "Outro"

    df["CAT_MACRO"] = df["CAT_RAW"].map(map_cat).fillna("Outro")

    # Modalidade
    if "TP_MODALIDADE_ENSINO" in df.columns:
        mod_map = {1: "Presencial", 2: "EAD", "1": "Presencial", "2": "EAD"}
        df["MODALIDADE"] = df["TP_MODALIDADE_ENSINO"].map(mod_map).fillna("Outro")
    else:
        df["MODALIDADE"] = "Outro"

    # Escopo (BR vs RJ)
    def map_escopo(row):
        if _norm(row.get("NO_MUNICIPIO", "")) == "RIO DE JANEIRO" and str(row.get("SG_UF", "")).upper() == "RJ":
            return "RJ"
        return "BR"

    df["ESCOPO"] = df.apply(map_escopo, axis=1)

    # tipagens finais
    for col in ["ANO", "VAGAS", "QT_ING", "QT_MAT", "QT_CONC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


master = load_data()
if master.empty:
    st.stop()

yr_min, yr_max = master["ANO"].min(), master["ANO"].max()


# -----------------------------
# SLICES / AGG
# -----------------------------
def slice_categoria(df: pd.DataFrame, escopo="BR"):
    view = (
        df.loc[df["ESCOPO"] == escopo, ["ANO", "CAT_MACRO", "QT_ING", "QT_MAT", "QT_CONC"]]
        .groupby(["ANO", "CAT_MACRO"], observed=True, as_index=False)
        .sum(min_count=1)
        .sort_values(["ANO", "CAT_MACRO"])
    )
    return view


def slice_modalidade(df: pd.DataFrame, escopo="BR"):
    view = (
        df.loc[df["ESCOPO"] == escopo, ["ANO", "MODALIDADE", "QT_ING", "QT_MAT", "QT_CONC"]]
        .groupby(["ANO", "MODALIDADE"], observed=True, as_index=False)
        .sum(min_count=1)
        .sort_values(["ANO", "MODALIDADE"])
    )
    return view


def compute_growth_table(df_long, group_col, value_col, a0, a1):
    base = (df_long[df_long["ANO"] == a0].set_index(group_col)[value_col].rename("base")).to_frame()
    end = (df_long[df_long["ANO"] == a1].set_index(group_col)[value_col].rename("end")).to_frame()
    tb = base.join(end, how="outer").fillna(0)
    years = max(1, a1 - a0)
    tb["Δ absoluto"] = (tb["end"] - tb["base"]).astype(float)
    tb["CAGR"] = np.where(
        (tb["base"] > 0) & (years >= 1),
        (tb["end"] / tb["base"]).replace([np.inf, -np.inf], np.nan) ** (1 / years) - 1,
        np.nan,
    )
    return tb.reset_index().sort_values("Δ absoluto", ascending=False)


# -----------------------------
# CHARTS
# -----------------------------
def line_with_labels(df, x, y, series, color_map, title, subtitle, yaxis_title, show_labels=True):
    fig = go.Figure()
    for key, sub in df.groupby(series):
        sub = sub.sort_values(x)
        fig.add_trace(
            go.Scatter(
                x=sub[x],
                y=sub[y],
                mode="lines+markers" + ("+text" if show_labels else ""),
                name=str(key),
                text=[format_short(v) for v in sub[y]] if show_labels else None,
                textposition="top center",
                textfont=dict(family=FONT_FAMILY, size=12),
                line=dict(color=color_map.get(str(key), PALETTE["azuis"][2]), width=3),
                marker=dict(size=7),
                cliponaxis=False,
                hovertemplate=f"{series}: {key}<br>{x}: %{{x}}<br>{yaxis_title}: %{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>{title}</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>{subtitle}</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90,
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=95, b=70),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
    )
    # Só grade horizontal
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR,
                     title=yaxis_title, tickformat=",.0f")
    return fig


def stacked_modalidade(df_mod, metric_col, title, subtitle, pct=False):
    pivot = df_mod.pivot(index="ANO", columns="MODALIDADE", values=metric_col).fillna(0).sort_index()
    # garantir colunas na ordem
    for col in ["Presencial", "EAD"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Presencial", "EAD"]]

    if pct:
        totals = pivot.sum(axis=1).replace(0, np.nan)
        ydata = (pivot.div(totals, axis=0) * 100).fillna(0)
        text_fmt = ydata.round(1).astype(str) + "%"
        yaxis_title = "Participação (%)"
        hover_y = "%{y:.1f}%"
        text_inside = True
    else:
        ydata = pivot
        text_fmt = ydata.applymap(format_short)
        yaxis_title = "Total"
        hover_y = "%{y:,.0f}"
        text_inside = False

    fig = go.Figure()
    for mod in ["Presencial", "EAD"]:
        fig.add_trace(
            go.Bar(
                x=ydata.index,
                y=ydata[mod],
                name=mod,
                marker_color=COLOR_MOD.get(mod, PALETTE["azuis"][2]),
                text=text_fmt[mod],
                textposition="inside" if text_inside else "auto",
                hovertemplate=f"Modalidade: {mod}<br>Ano: %{{x}}<br>Valor: {hover_y}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b style='color:{TITLE_COLOR}'>{title}</b>"
                 f"<br><span style='color:{SUBTITLE_COLOR}; font-weight: normal;'>{subtitle}</span>",
            font=dict(family=FONT_FAMILY, size=20),
            x=0, xanchor="left", y=0.90
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color=AXIS_COLOR),
        margin=dict(l=10, r=10, t=120, b=60),
    )
    # Só grade horizontal
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_COLOR, tickmode="linear")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     linecolor=AXIS_COLOR, title=yaxis_title,
                     tickformat=None if pct else ",.0f", rangemode="tozero")
    return fig


# -----------------------------
# UI
# -----------------------------
st.markdown(
    f"<h2 style='color:{TITLE_COLOR};font-family:{FONT_FAMILY};font-weight:700;margin-bottom:0'>"
    f"Administração (Bacharelado) — INEP {yr_min}–{yr_max}</h2>"
    f"<p style='color:{SUBTITLE_COLOR};font-family:{FONT_FAMILY};margin-top:2px'>"
    f"Comparativos: Público vs Privado e Presencial vs EAD • Brasil e RJ • "
    f"Métricas: Ingressantes, Matrículas, Concluintes</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Parâmetros")
    anos_sel = st.slider("Período", min_value=int(yr_min), max_value=int(yr_max),
                         value=(int(yr_min), int(yr_max)), step=1)
    metrica = st.radio("Métrica", ["Matrículas", "Concluintes", "Ingressantes"], index=0, horizontal=True)
    view = st.radio("Visualização", ["Por categoria administrativa", "Por modalidade"], index=0)
    empilhado_pct = st.checkbox("Market Share 100% (participação)", value=True)
    show_labels = st.checkbox("Mostrar rótulos nos pontos (linhas)", value=True)
    mostrar_tabelas = st.checkbox("Mostrar tabelas resumo", value=False)
    st.caption("Dica: senha via `st.secrets['password']` ou variável de ambiente `APP_PASSWORD`.")

yr0, yr1 = anos_sel
metric_map = {"Matrículas": "QT_MAT", "Concluintes": "QT_CONC", "Ingressantes": "QT_ING"}
metric_col = metric_map[metrica]
m = master[(master["ANO"] >= yr0) & (master["ANO"] <= yr1)].copy()


# -----------------------------
# RENDER
# -----------------------------
def render_scope(scope_label: str, escopo_df: pd.DataFrame):
    col1, col2 = st.columns([2, 1], gap="large")

    if view == "Por categoria administrativa":
        df_cat = slice_categoria(escopo_df, escopo=scope_label)
        fig = line_with_labels(
            df_cat, x="ANO", y=metric_col, series="CAT_MACRO",
            color_map=COLOR_CAT,
            title=f"{metrica} por Categoria Administrativa",
            subtitle=f"Administração (Bacharelado) • {scope_label} • {yr0}–{yr1}",
            yaxis_title=metrica, show_labels=show_labels,
        )
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_cat.rename(columns={"CAT_MACRO": "GRUPO"}), "GRUPO", metric_col, yr0, yr1)

    else:
        df_mod = slice_modalidade(escopo_df, escopo=scope_label)
        fig = line_with_labels(
            df_mod, x="ANO", y=metric_col, series="MODALIDADE",
            color_map=COLOR_MOD,
            title=f"{metrica} por Modalidade",
            subtitle=f"Administração (Bacharelado) • {scope_label} • {yr0}–{yr1}",
            yaxis_title=metrica, show_labels=show_labels,
        )
        col1.plotly_chart(fig, use_container_width=True)
        tb = compute_growth_table(df_mod.rename(columns={"MODALIDADE": "GRUPO"}), "GRUPO", metric_col, yr0, yr1)

    if not tb.empty and mostrar_tabelas:
        tb["Δ absoluto"] = tb["Δ absoluto"].round(0).astype(int)
        tb["CAGR (%)"] = (tb["CAGR"] * 100).round(2)
        tb_display = tb[["GRUPO", "base", "end", "Δ absoluto", "CAGR (%)"]]
        col2.subheader("Crescimento no período")
        col2.dataframe(tb_display, use_container_width=True, hide_index=True)

    # Stacked / Market share
    st.markdown("### Market share por Modalidade")
    df_mod2 = slice_modalidade(escopo_df, escopo=scope_label)
    st.plotly_chart(
        stacked_modalidade(
            df_mod2, metric_col,
            title=f"{metrica} por Modalidade (Empilhado)",
            subtitle=f"{scope_label} • {yr0}–{yr1}", pct=False,
        ),
        use_container_width=True,
    )
    if empilhado_pct:
        st.plotly_chart(
            stacked_modalidade(
                df_mod2, metric_col,
                title="Participação (%) por Modalidade (100%)",
                subtitle=f"{scope_label} • {yr0}–{yr1}", pct=True,
            ),
            use_container_width=True,
        )


tab_br, tab_rj = st.tabs(["🇧🇷 Brasil", "🏷️ RJ"])
with tab_br:
    render_scope("BR", m)
with tab_rj:
    render_scope("RJ", m)
