# GestãoDS — ICP & Pricing Intelligence Pipeline

Pipeline completo de Machine Learning para identificação do ICP, modelagem de churn,
segmentação de clientes e scores de propensão para cross-sell.

---

## 📁 Estrutura do Projeto

```
gestaods_icp/
│
├── data/
│   └── clientes.csv              ← SEU ARQUIVO DE ENTRADA (coloque aqui)
│
├── src/
│   ├── 01_preprocessing.py       ← Limpeza, encoding, feature engineering
│   ├── 02_ltv_analysis.py        ← LTV, coortes, top 25%, análise por canal
│   ├── 03_segmentation.py        ← Clustering K-Means (segmentos ICP)
│   ├── 04_churn_model.py         ← Modelo XGBoost de churn + scores de risco
│   ├── 05_propensity_models.py   ← Modelos de propensão: DS Pay, IA, ChatGDS
│   └── 06_report.py              ← Geração do relatório Excel
│
├── outputs/
│   ├── clientes_scored.csv       ← Base enriquecida com todos os scores
│   ├── icp_report.xlsx           ← Relatório Excel (11 abas)
│   ├── models/                   ← Modelos treinados (.pkl)
│   └── *.png                     ← Visualizações geradas
│
├── app.py                        ← Dashboard Streamlit (5 abas interativas)
├── run_pipeline.py               ← Orquestrador: executa tudo em sequência
└── requirements.txt
```

---

## 🚀 Como Usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Colocar o CSV
Coloque seu arquivo CSV em `data/clientes.csv`.

O pipeline aceita os nomes de colunas exatos ou variações (case-insensitive).
Se alguma coluna tiver nome diferente do esperado, edite o `COLUMN_MAP`
no início do arquivo `src/01_preprocessing.py`.

### 3. Executar o pipeline completo
```bash
python run_pipeline.py
```

### 4. Rodar só a análise (sem modelos ML)
```bash
python run_pipeline.py --skip-models
```

### 5. Abrir o dashboard interativo
```bash
streamlit run app.py
```

---

## 📊 O que o Pipeline Entrega

### Arquivo: `outputs/clientes_scored.csv`
Base original enriquecida com:
- `ltv_realizado` e `ltv_projetado`
- `ltv_quartil` e `top25_ltv` (flag dos 25% com maior LTV)
- `segmento` e `segmento_rotulo` (cluster ICP)
- `score_churn` (0–100) e `risco_churn` (Alto/Médio/Baixo)
- `score_propensao_ds_pay` (0–100)
- `score_propensao_ia` (0–100)
- `score_propensao_chatgds` (0–100)
- `score_adocao_composto` (métrica derivada de uso)
- `faturamento_reconciliado` (faturamento validado/corrigido)

### Arquivo: `outputs/icp_report.xlsx` (11 abas)
1. **Resumo Executivo** — KPIs gerais e alertas do pipeline
2. **Top 25% ICP** — Perfil detalhado dos melhores clientes
3. **Base Completa Scored** — Todos os clientes com scores
4. **Segmentos ICP** — Perfil dos clusters
5. **Risco de Churn** — Clientes ativos ordenados por risco
6. **Propensão DS Pay** — Top 50 para abordagem
7. **Propensão Agentes IA** — Top 50 para abordagem
8. **Propensão ChatGDS** — Top 50 para abordagem
9. **Canal — Base completa** — LTV/churn por canal
10. **Canal — Rastreável** — Análise sem viés de canal nulo
11. **Coorte de Retenção** — Retenção por trimestre de entrada
12. **Tiers de Precificação** — Base para nova estrutura de planos

### Dashboard Streamlit (5 abas)
- **Visão Geral ICP**: LTV por especialidade, porte, mapa geográfico
- **Risco de Churn**: Score de risco, lista de ação, análise por segmento
- **Propensão**: Rankings por produto com filtros interativos
- **Financeiro & LTV**: Distribuições, canais, tiers de precificação
- **Segmentos**: Drill-down interativo por cluster

---

## 🔧 Customizações Importantes

### Nomes de colunas do seu CSV
Edite o `COLUMN_MAP` em `src/01_preprocessing.py`:
```python
COLUMN_MAP = {
    "id_cliente": "ID_Cliente",    # ← mude o valor se o CSV tiver nome diferente
    "mrr_atual":  "MRR_Geral_Atual",
    # ...
}
```

### Número de clusters
Por padrão o pipeline escolhe automaticamente pelo silhouette score.
Para forçar um k específico, edite `src/03_segmentation.py`:
```python
k = find_optimal_k(X)  # automático
# ou
k = 5  # forçado
```

### Threshold de risco de churn
Ajuste os bins em `src/04_churn_model.py`:
```python
df["risco_churn"] = pd.cut(
    df["score_churn"].fillna(0),
    bins=[-1, 30, 60, 101],           # ← ajuste os limites aqui
    labels=["Baixo", "Médio", "Alto"],
)
```

---

## ⚠️ Pontos de Atenção

**Canal de aquisição ausente:**
Clientes sem canal são categorizados como "nao_rastreado" e incluídos na análise
geral, mas separados na análise de ROI por canal para evitar viés.

**NPS e data leakage:**
O pipeline testa automaticamente se o NPS disponível tem correlação forte com churn
(sinal de que foi coletado no cancelamento). Se sim, o NPS é excluído do modelo
preditivo mas mantido nas análises descritivas.

**Faturamento estimado:**
O pipeline calcula uma estimativa alternativa baseada em Ticket_Medio × Pacientes_Ativos
e reconcilia automaticamente quando há divergência > 40%.

**Base mínima para modelos supervisionados:**
- Modelo de churn: mínimo 100 cancelados na base
- Modelos de propensão: mínimo 20 adotantes de cada produto

---

## 📦 Dependências

```
pandas, numpy, scikit-learn, xgboost, imbalanced-learn,
matplotlib, seaborn, plotly, streamlit, openpyxl, joblib, scipy
```

```bash
pip install -r requirements.txt
```

---

*Pipeline desenvolvido para análise estratégica de ICP e precificação da GestãoDS.*
