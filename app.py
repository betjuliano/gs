"""
app.py — Dashboard Streamlit
GestãoDS ICP & Pricing Intelligence

Abas:
  1. 📊 Visão Geral ICP
  2. 🔴 Risco de Churn
  3. 🚀 Propensão — Cross-sell
  4. 💰 Análise Financeira & LTV
  5. 🔬 Segmentos

Uso: streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GestãoDS — ICP Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 16px 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h3 { font-size: 28px; margin: 0; font-weight: 700; }
    .metric-card p  { font-size: 12px; margin: 4px 0 0; opacity: 0.85; }

    .risk-alto   { background: #FEE2E2; border-left: 4px solid #EF4444; padding: 8px; border-radius: 4px; }
    .risk-medio  { background: #FEF9C3; border-left: 4px solid #F59E0B; padding: 8px; border-radius: 4px; }
    .risk-baixo  { background: #DCFCE7; border-left: 4px solid #22C55E; padding: 8px; border-radius: 4px; }

    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 26px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    scored_path = "outputs/clientes_scored.csv"
    if os.path.exists(scored_path):
        return pd.read_csv(scored_path, encoding="utf-8-sig", low_memory=False)

    # Fallback: rodar pipeline on-the-fly
    raw_path = "data/clientes.csv"
    if not os.path.exists(raw_path):
        return pd.DataFrame()

    from src.preprocessing_01 import run_preprocessing
    from src.ltv_analysis_02 import run_ltv_analysis
    df, meta = run_preprocessing(raw_path)
    df, _    = run_ltv_analysis(df)
    return df


def safe_col(df, col, default=0):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBAIS
# ─────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.image("https://via.placeholder.com/200x60?text=GestãoDS", width=200)
    st.sidebar.markdown("## 🔍 Filtros Globais")

    filtered = df.copy()

    # Status
    if "churn" in df.columns:
        status_opts = {"Todos": None, "Apenas Ativos": 0, "Apenas Cancelados": 1}
        status_sel  = st.sidebar.selectbox("Status do Cliente", list(status_opts.keys()))
        if status_opts[status_sel] is not None:
            filtered = filtered[filtered["churn"] == status_opts[status_sel]]

    # UF
    if "uf" in df.columns:
        ufs = sorted(df["uf"].dropna().unique().tolist())
        sel_ufs = st.sidebar.multiselect("Estado (UF)", ufs, default=[])
        if sel_ufs:
            filtered = filtered[filtered["uf"].isin(sel_ufs)]

    # Especialidade
    if "especialidade_principal" in df.columns:
        esps = sorted(df["especialidade_principal"].dropna().unique().tolist())
        sel_esps = st.sidebar.multiselect("Especialidade Principal", esps, default=[])
        if sel_esps:
            filtered = filtered[filtered["especialidade_principal"].isin(sel_esps)]

    # Porte
    if "porte_clinica" in df.columns:
        portes = df["porte_clinica"].dropna().astype(str).unique().tolist()
        sel_portes = st.sidebar.multiselect("Porte da Clínica", portes, default=[])
        if sel_portes:
            filtered = filtered[filtered["porte_clinica"].astype(str).isin(sel_portes)]

    # Canal
    if "canal_origem" in df.columns:
        canais = sorted(df["canal_origem"].dropna().unique().tolist())
        sel_canais = st.sidebar.multiselect("Canal de Aquisição", canais, default=[])
        if sel_canais:
            filtered = filtered[filtered["canal_origem"].isin(sel_canais)]

    # MRR range
    if "mrr_atual" in df.columns:
        mrr_min = float(df["mrr_atual"].min() or 0)
        mrr_max = float(df["mrr_atual"].max() or 10000)
        mrr_range = st.sidebar.slider("MRR Atual (R$)", mrr_min, mrr_max,
                                       (mrr_min, mrr_max))
        filtered = filtered[
            filtered["mrr_atual"].fillna(0).between(mrr_range[0], mrr_range[1])
        ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(filtered):,} clientes** com filtros aplicados")
    return filtered


# ─────────────────────────────────────────────────────────────
# MÉTRICAS KPI
# ─────────────────────────────────────────────────────────────
def kpi_row(df: pd.DataFrame):
    cols = st.columns(5)
    n_total   = len(df)
    n_ativos  = (safe_col(df, "churn") == 0).sum()
    churn_pct = round(safe_col(df, "churn").mean() * 100, 1)
    mrr_total = safe_col(df, "mrr_atual").fillna(0).sum()
    ltv_med   = safe_col(df, "ltv_realizado").median()

    with cols[0]:
        st.metric("👥 Total Clientes",  f"{n_total:,}")
    with cols[1]:
        st.metric("✅ Ativos",          f"{n_ativos:,}")
    with cols[2]:
        st.metric("📉 Churn Rate",      f"{churn_pct}%",
                  delta=f"{churn_pct - 100:.1f}% da retenção")
    with cols[3]:
        st.metric("💵 MRR Total",       f"R$ {mrr_total:,.0f}")
    with cols[4]:
        st.metric("📈 LTV Mediano",     f"R$ {ltv_med:,.0f}" if not np.isnan(ltv_med) else "—")


# ─────────────────────────────────────────────────────────────
# ABA 1 — VISÃO GERAL ICP
# ─────────────────────────────────────────────────────────────
def tab_icp_overview(df: pd.DataFrame):
    st.header("📊 Visão Geral do ICP")
    kpi_row(df)
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        if "especialidade_principal" in df.columns and "ltv_realizado" in df.columns:
            esp_ltv = (
                df.groupby("especialidade_principal")["ltv_realizado"]
                .median()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            fig = px.bar(esp_ltv, x="ltv_realizado", y="especialidade_principal",
                         orientation="h",
                         title="LTV Mediano por Especialidade (Top 15)",
                         labels={"ltv_realizado": "LTV Mediano (R$)",
                                  "especialidade_principal": ""},
                         color="ltv_realizado",
                         color_continuous_scale="Blues")
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "porte_clinica" in df.columns and "churn" in df.columns:
            porte_data = (
                df.groupby("porte_clinica", observed=True)
                .agg(n=("churn", "count"),
                     churn_rate=("churn", "mean"),
                     mrr_med=("mrr_atual", "median"))
                .reset_index()
            )
            porte_data["churn_pct"] = (porte_data["churn_rate"] * 100).round(1)
            fig = px.scatter(porte_data, x="mrr_med", y="churn_pct",
                             size="n", color="porte_clinica",
                             title="MRR × Churn Rate por Porte",
                             labels={"mrr_med": "MRR Mediano (R$)",
                                      "churn_pct": "Churn Rate (%)",
                                      "porte_clinica": "Porte"},
                             size_max=50)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    # Top 25% ICP
    if "top25_ltv" in df.columns:
        st.subheader("🏆 Top 25% por LTV")
        top = df[df["top25_ltv"] == 1].copy()

        cols_show = [c for c in [
            "nome_clinica", "especialidade_principal", "uf",
            "porte_clinica", "mrr_atual", "ltv_realizado",
            "score_adocao_composto", "dias_ativo", "risco_churn",
        ] if c in top.columns]

        st.dataframe(
            top[cols_show].sort_values("ltv_realizado", ascending=False)
            .reset_index(drop=True).head(100),
            use_container_width=True,
            height=350,
        )

    # Distribuição por UF
    if "uf" in df.columns:
        uf_count = df["uf"].value_counts().reset_index()
        uf_count.columns = ["UF", "Clientes"]
        fig = px.choropleth(uf_count,
                             locations="UF",
                             locationmode="geojson-id",
                             color="Clientes",
                             scope="south america",
                             title="Distribuição Geográfica de Clientes",
                             color_continuous_scale="Blues")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ABA 2 — RISCO DE CHURN
# ─────────────────────────────────────────────────────────────
def tab_churn(df: pd.DataFrame):
    st.header("🔴 Análise de Risco de Churn")

    ativos = df[safe_col(df, "churn") == 0].copy()

    if "risco_churn" not in ativos.columns:
        st.warning("Modelo de churn não foi executado. Rode o pipeline completo.")
        return

    # KPIs de risco
    c1, c2, c3 = st.columns(3)
    n_alto  = (ativos["risco_churn"] == "Alto").sum()
    n_medio = (ativos["risco_churn"] == "Médio").sum()
    n_baixo = (ativos["risco_churn"] == "Baixo").sum()
    mrr_risco = ativos[ativos["risco_churn"] == "Alto"]["mrr_atual"].fillna(0).sum()

    c1.metric("🔴 Alto Risco",  f"{n_alto:,} clientes")
    c2.metric("🟡 Médio Risco", f"{n_medio:,} clientes")
    c3.metric("💸 MRR em Alto Risco", f"R$ {mrr_risco:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        dist = ativos["risco_churn"].value_counts().reset_index()
        dist.columns = ["Risco", "Clientes"]
        fig = px.pie(dist, values="Clientes", names="Risco",
                     title="Distribuição de Risco de Churn",
                     color="Risco",
                     color_discrete_map={"Alto": "#EF4444",
                                          "Médio": "#F59E0B",
                                          "Baixo": "#22C55E"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "score_churn" in ativos.columns:
            fig = px.histogram(ativos, x="score_churn", nbins=30,
                                title="Distribuição Score de Churn",
                                color_discrete_sequence=["#F97316"])
            fig.add_vline(x=60, line_dash="dash", line_color="red",
                          annotation_text="Threshold Alto")
            fig.add_vline(x=30, line_dash="dash", line_color="orange",
                          annotation_text="Threshold Médio")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # Tabela de alto risco
    st.subheader("🚨 Clientes em Alto Risco — Prioridade de Ação")
    alto = ativos[ativos["risco_churn"] == "Alto"].copy()

    cols_show = [c for c in [
        "nome_clinica", "especialidade_principal", "uf",
        "mrr_atual", "score_churn", "score_adocao_composto",
        "maturidade_meses", "segmento_rotulo",
    ] if c in alto.columns]

    st.dataframe(
        alto[cols_show].sort_values("score_churn", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # Churn por segmento
    if "segmento_rotulo" in ativos.columns and "score_churn" in ativos.columns:
        seg_churn = (
            ativos.groupby("segmento_rotulo")["score_churn"]
            .mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(seg_churn, x="segmento_rotulo", y="score_churn",
                     title="Score Médio de Churn por Segmento",
                     color="score_churn",
                     color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ABA 3 — PROPENSÃO
# ─────────────────────────────────────────────────────────────
def tab_propensity(df: pd.DataFrame):
    st.header("🚀 Propensão para Cross-sell / Upsell")

    products = {
        "DS Pay":      ("score_propensao_ds_pay",   "usa_conciliacao", "#7C3AED"),
        "Agentes IA":  ("score_propensao_ia",        "usa_ia",          "#0891B2"),
        "ChatGDS":     ("score_propensao_chatgds",   "usa_chatgds",     "#16A34A"),
    }

    tabs = st.tabs(list(products.keys()))
    ativos = df[safe_col(df, "churn") == 0].copy()

    for tab, (nome, (score_col, label_col, color)) in zip(tabs, products.items()):
        with tab:
            if score_col not in ativos.columns:
                st.info(f"Score de propensão para {nome} ainda não calculado.")
                continue

            sem_prod = ativos[safe_col(ativos, label_col) == 0]
            com_prod = ativos[safe_col(ativos, label_col) == 1]

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Sem {nome}", f"{len(sem_prod):,}")
            c2.metric(f"Já usa {nome}", f"{len(com_prod):,}")
            c3.metric("Score Mediano",
                       f"{sem_prod[score_col].median():.0f}/100"
                       if len(sem_prod) > 0 else "—")

            col1, col2 = st.columns(2)

            with col1:
                if len(sem_prod) > 0:
                    fig = px.histogram(sem_prod, x=score_col, nbins=25,
                                        title=f"Distribuição — Propensão {nome}",
                                        color_discrete_sequence=[color])
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "especialidade_principal" in sem_prod.columns and len(sem_prod) > 0:
                    esp_score = (
                        sem_prod.groupby("especialidade_principal")[score_col]
                        .mean().sort_values(ascending=False).head(10).reset_index()
                    )
                    fig = px.bar(esp_score, x=score_col, y="especialidade_principal",
                                  orientation="h",
                                  title=f"Top Especialidades — {nome}",
                                  color_discrete_sequence=[color])
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            # Lista de prioridade
            if len(sem_prod) > 0:
                st.subheader(f"🎯 Top 50 Clientes para {nome}")
                cols_show = [c for c in [
                    "nome_clinica", "especialidade_principal", "uf",
                    "mrr_atual", score_col, "score_adocao_composto",
                    "maturidade_meses", "segmento_rotulo",
                ] if c in sem_prod.columns]

                top50 = (sem_prod[cols_show]
                          .sort_values(score_col, ascending=False)
                          .head(50).reset_index(drop=True))
                top50.index += 1
                st.dataframe(top50, use_container_width=True, height=380)


# ─────────────────────────────────────────────────────────────
# ABA 4 — FINANCEIRO & LTV
# ─────────────────────────────────────────────────────────────
def tab_financial(df: pd.DataFrame):
    st.header("💰 Análise Financeira & LTV")

    col1, col2 = st.columns(2)

    with col1:
        if "ltv_realizado" in df.columns:
            fig = px.histogram(df, x="ltv_realizado",
                                nbins=50,
                                title="Distribuição LTV Realizado",
                                color_discrete_sequence=["#2563EB"])
            q75 = df["ltv_realizado"].quantile(0.75)
            fig.add_vline(x=q75, line_dash="dash", line_color="green",
                          annotation_text=f"Top 25%: R${q75:,.0f}")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "mrr_atual" in df.columns and "score_adocao_composto" in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))),
                              x="score_adocao_composto",
                              y="mrr_atual",
                              color=safe_col(df, "risco_churn").reindex(
                                  df.sample(min(1000, len(df))).index
                              ) if "risco_churn" in df.columns else None,
                              title="Adoção × MRR",
                              labels={"score_adocao_composto": "Score de Adoção",
                                       "mrr_atual": "MRR (R$)"},
                              opacity=0.6,
                              height=380)
            st.plotly_chart(fig, use_container_width=True)

    # LTV por canal
    if "canal_origem" in df.columns and "ltv_realizado" in df.columns:
        canal_ltv = (
            df.groupby("canal_origem")
            .agg(ltv_med=("ltv_realizado", "median"),
                  n=("ltv_realizado", "count"),
                  churn_pct=("churn", "mean"))
            .reset_index()
        )
        canal_ltv["churn_pct"] = (canal_ltv["churn_pct"] * 100).round(1)
        canal_ltv = canal_ltv.sort_values("ltv_med", ascending=False)

        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=["LTV Mediano por Canal",
                                              "Churn Rate % por Canal"])
        fig.add_trace(
            go.Bar(x=canal_ltv["canal_origem"], y=canal_ltv["ltv_med"],
                   marker_color="#2563EB", name="LTV"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=canal_ltv["canal_origem"], y=canal_ltv["churn_pct"],
                   marker_color="#EF4444", name="Churn %"),
            row=1, col=2
        )
        fig.update_layout(height=380, title_text="Análise por Canal de Aquisição",
                           showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Tiers de precificação
    if "faixa_pacientes" in df.columns:
        st.subheader("💡 Tiers de Precificação por Base de Pacientes")
        tiers = (
            df.groupby("faixa_pacientes", observed=True)
            .agg(n=("mrr_atual", "count"),
                  mrr_med=("mrr_atual", "median"),
                  ltv_med=("ltv_realizado", "median"),
                  churn_pct=("churn", "mean"))
            .reset_index()
        )
        tiers["churn_pct"] = (tiers["churn_pct"] * 100).round(1)
        st.dataframe(tiers, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ABA 5 — SEGMENTOS
# ─────────────────────────────────────────────────────────────
def tab_segments(df: pd.DataFrame):
    st.header("🔬 Segmentos ICP")

    if "segmento_rotulo" not in df.columns:
        st.warning("Segmentação não executada. Rode o pipeline completo.")
        return

    seg_summary = (
        df.groupby("segmento_rotulo")
        .agg(n=("churn", "count"),
              mrr_med=("mrr_atual", "median"),
              ltv_med=("ltv_realizado", "median"),
              churn_pct=("churn", "mean"),
              score_adocao=("score_adocao_composto", "median"),
              maturidade_med=("maturidade_meses", "median"))
        .reset_index()
    )
    seg_summary["churn_pct"] = (seg_summary["churn_pct"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(seg_summary,
                          x="mrr_med", y="churn_pct",
                          size="n", color="segmento_rotulo",
                          title="Segmentos: MRR × Churn",
                          labels={"mrr_med": "MRR Mediano (R$)",
                                   "churn_pct": "Churn Rate (%)",
                                   "segmento_rotulo": "Segmento"},
                          size_max=60, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(seg_summary.sort_values("ltv_med", ascending=False),
                      x="segmento_rotulo", y="ltv_med",
                      title="LTV Mediano por Segmento",
                      color="ltv_med",
                      color_continuous_scale="Greens",
                      height=420)
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Perfil dos Segmentos")
    st.dataframe(seg_summary.sort_values("ltv_med", ascending=False),
                  use_container_width=True)

    # Drill-down por segmento
    seg_sel = st.selectbox("Explorar segmento",
                             df["segmento_rotulo"].dropna().unique())
    if seg_sel:
        sub = df[df["segmento_rotulo"] == seg_sel]
        cols_show = [c for c in [
            "nome_clinica", "especialidade_principal", "uf",
            "mrr_atual", "ltv_realizado", "score_adocao_composto",
            "risco_churn", "dias_ativo",
        ] if c in sub.columns]
        st.dataframe(sub[cols_show].reset_index(drop=True),
                      use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🏥 GestãoDS — ICP & Pricing Intelligence")
    st.markdown("*Análise estratégica de clientes, churn e oportunidades de expansão*")

    df = load_data()

    if df.empty:
        st.error("""
        ❌ Base de dados não encontrada.

        **Para iniciar:**
        1. Coloque seu CSV em `data/clientes.csv`
        2. Execute: `python run_pipeline.py`
        3. Recarregue esta página
        """)
        return

    df_filtered = render_sidebar(df)

    if len(df_filtered) == 0:
        st.warning("Nenhum cliente corresponde aos filtros selecionados.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral ICP",
        "🔴 Risco de Churn",
        "🚀 Propensão",
        "💰 Financeiro & LTV",
        "🔬 Segmentos",
    ])

    with tab1:
        tab_icp_overview(df_filtered)
    with tab2:
        tab_churn(df_filtered)
    with tab3:
        tab_propensity(df_filtered)
    with tab4:
        tab_financial(df_filtered)
    with tab5:
        tab_segments(df_filtered)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#9CA3AF; font-size:12px'>"
        "GestãoDS ICP Intelligence — Pipeline de Machine Learning"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
