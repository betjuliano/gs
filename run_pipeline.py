"""
run_pipeline.py — Orquestrador Principal
GestãoDS ICP Analysis Pipeline

Uso:
    python run_pipeline.py                         # usa data/clientes.csv
    python run_pipeline.py data/minha_base.csv     # caminho customizado
    python run_pipeline.py --skip-models           # pula modelos ML (só análise)
"""

import sys
import os
import time
import argparse
import pandas as pd

# Garante que o diretório src está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(texto: str):
    print("\n" + "█" * 62)
    print(f"  {texto}")
    print("█" * 62)


def parse_args():
    parser = argparse.ArgumentParser(description="GestãoDS ICP Pipeline")
    parser.add_argument("filepath", nargs="?", default="data/clientes.csv",
                        help="Caminho para o CSV de entrada")
    parser.add_argument("--skip-models", action="store_true",
                        help="Pula treinamento de modelos ML")
    return parser.parse_args()


def main():
    args     = parse_args()
    filepath = args.filepath
    t_start  = time.time()

    banner("GestãoDS — ICP & Pricing Intelligence Pipeline")
    print(f"  Arquivo : {filepath}")
    print(f"  Modelos : {'DESATIVADOS' if args.skip_models else 'ATIVADOS'}")

    if not os.path.exists(filepath):
        print(f"\n❌ Arquivo não encontrado: {filepath}")
        print("   Coloque seu CSV em data/clientes.csv e execute novamente.")
        sys.exit(1)

    # ── MÓDULO 01: Pré-processamento ─────────────────────────
    from src.preprocessing_01 import run_preprocessing
    df, meta = run_preprocessing(filepath)

    # ── MÓDULO 02: LTV e Coortes ─────────────────────────────
    from src.ltv_analysis_02 import run_ltv_analysis
    df, ltv_results = run_ltv_analysis(df)

    results = ltv_results.copy()

    # ── MÓDULO 03: Segmentação ICP ───────────────────────────
    perfil_df = None
    km_model  = None

    if not args.skip_models:
        from src.segmentation_03 import run_segmentation
        df, perfil_df, km_model = run_segmentation(df)
    else:
        print("\n⏭️  Segmentação: pulado (--skip-models)")

    # ── MÓDULO 04: Modelo de Churn ───────────────────────────
    if not args.skip_models:
        from src.churn_model_04 import run_churn_model
        df, churn_model, churn_features, threshold, cv_scores = run_churn_model(
            df, meta["nps_audit"]
        )
        results["churn_model"] = {
            "auc_mean": cv_scores.mean(),
            "threshold": threshold,
        }
    else:
        print("\n⏭️  Modelo de churn: pulado (--skip-models)")

    # ── MÓDULO 05: Propensão ─────────────────────────────────
    priority_lists = {}
    if not args.skip_models:
        from src.propensity_models_05 import run_propensity_models
        df, priority_lists, prop_models = run_propensity_models(df)
    else:
        print("\n⏭️  Modelos de propensão: pulados (--skip-models)")

    # ── MÓDULO 06: Relatório Excel ───────────────────────────
    from src.report_06 import run_report
    report_path = run_report(df, results, meta, perfil_df, priority_lists)

    # ── SALVA BASE ENRIQUECIDA ────────────────────────────────
    scored_path = "outputs/clientes_scored.csv"
    df.to_csv(scored_path, index=False, encoding="utf-8-sig")
    print(f"✅ Base enriquecida salva: {scored_path}")

    # ── SUMÁRIO FINAL ─────────────────────────────────────────
    elapsed = time.time() - t_start
    banner("PIPELINE CONCLUÍDO")
    print(f"  Tempo total   : {elapsed:.1f}s")
    print(f"  Clientes      : {meta['n_total']:,}")
    print(f"  Relatório     : {report_path}")
    print(f"  Base scored   : {scored_path}")
    print(f"  Visualizações : outputs/*.png")
    if not args.skip_models and "churn_model" in results:
        print(f"  AUC Churn     : {results['churn_model']['auc_mean']:.3f}")
    print(f"\n  Para visualizar o dashboard:")
    print(f"  👉  streamlit run app.py")
    print("█" * 62 + "\n")

    return df, results, meta


if __name__ == "__main__":
    main()
