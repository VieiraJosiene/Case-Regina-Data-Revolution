"""Case Regina - Data Revolution 2026 (UFJF) - Grupo 11.

Compara regressao logistica e arvore de decisao para ordenar clientes pela
probabilidade de compra e estima o resultado financeiro de uma campanha de
e-mail com orcamento limitado.

Uso:
    python src/analise_campanha.py
    python src/analise_campanha.py --data data/campanha_venda_model.csv \
        --drop-cols ID --engagement-col Engajamento
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Premissas informadas no desafio
ORCAMENTO = 40_000        # R$, cobre todos os gastos da campanha
CUSTO_EMAIL = 1           # R$ por e-mail
LUCRO_POR_VENDA = 100     # R$ por venda (confira com a apresentacao final)
BASE_CLIENTES = 1_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/campanha_venda_model.csv", help="CSV da campanha anterior")
    p.add_argument("--target", default="Campanha", help="coluna alvo (1 = comprou, 0 = nao comprou)")
    p.add_argument("--drop-cols", nargs="*", default=[], help="colunas a ignorar (ex.: identificadores)")
    p.add_argument("--engagement-col", default=None,
                   help="coluna de engajamento; se informada, roda tambem o cenario sem ela")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tree-depth", type=int, default=5, help="profundidade maxima da arvore")
    p.add_argument("--out", default="results/metricas.json")
    return p.parse_args()


def montar_modelos(seed: int, depth: int) -> dict:
    return {
        "Regressao logistica": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Arvore de decisao": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=seed),
        ),
    }


def metricas_topo(y_test: np.ndarray, score: np.ndarray, k: int) -> dict:
    """Resultado ao enviar e-mail apenas para os k clientes de maior score."""
    topo = np.argsort(-score)[:k]
    compras = int(y_test[topo].sum())
    return {
        "k_emails": int(k),
        "compradores_no_topo": compras,
        "taxa_no_topo": compras / k,
        "resultado_rs": compras * LUCRO_POR_VENDA - k * CUSTO_EMAIL,
    }


def cenario_campanha(taxa: float) -> dict:
    """Aplica a taxa de compra do topo aos e-mails que cabem no orcamento."""
    emails = int(ORCAMENTO / CUSTO_EMAIL)
    compras = taxa * emails
    return {
        "emails": emails,
        "taxa": taxa,
        "compras": round(compras),
        "resultado_rs": round(compras * LUCRO_POR_VENDA - ORCAMENTO),
    }


def avaliar(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    y_te = y_te.to_numpy()
    k = round(len(y_te) * ORCAMENTO / BASE_CLIENTES)  # 4% do teste

    saida = {"n_treino": len(y_tr), "n_teste": len(y_te), "taxa_geral": float(y.mean()), "modelos": {}}
    for nome, modelo in montar_modelos(args.seed, args.tree_depth).items():
        modelo.fit(X_tr, y_tr)
        score = modelo.predict_proba(X_te)[:, 1]
        saida["modelos"][nome] = {"auc": roc_auc_score(y_te, score), **metricas_topo(y_te, score, k)}

    aleatorio = float(y_te.mean())
    saida["aleatorio"] = {
        "k_emails": int(k),
        "compradores_esperados": aleatorio * k,
        "taxa_no_topo": aleatorio,
        "resultado_rs": aleatorio * k * LUCRO_POR_VENDA - k * CUSTO_EMAIL,
    }
    return saida


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, sep=None, engine="python")
    df = df.drop(columns=args.drop_cols)
    y = df.pop(args.target).astype(int)
    X = pd.get_dummies(df, drop_first=True)

    print(f"Base: {len(y):,} registros | taxa de compra: {y.mean():.1%}")
    resultado = {"com_engajamento": avaliar(X, y, args)}

    if args.engagement_col:
        X_sem = X.drop(columns=[c for c in X.columns if c.startswith(args.engagement_col)])
        resultado["sem_engajamento"] = avaliar(X_sem, y, args)

    # Cenarios para os 40 mil e-mails (modelo recomendado: regressao logistica)
    rl = "Regressao logistica"
    resultado["cenarios_40mil_emails"] = {
        "aleatorio": cenario_campanha(resultado["com_engajamento"]["aleatorio"]["taxa_no_topo"]),
        "modelo_com_engajamento": cenario_campanha(resultado["com_engajamento"]["modelos"][rl]["taxa_no_topo"]),
    }
    if "sem_engajamento" in resultado:
        resultado["cenarios_40mil_emails"]["modelo_sem_engajamento"] = cenario_campanha(
            resultado["sem_engajamento"]["modelos"][rl]["taxa_no_topo"]
        )

    for chave, bloco in resultado.items():
        if chave == "cenarios_40mil_emails":
            continue
        print(f"\n== {chave.replace('_', ' ')} (teste: {bloco['n_teste']:,} clientes) ==")
        for nome, m in bloco["modelos"].items():
            print(f"{nome:22s} AUC {m['auc']:.3f} | topo {m['k_emails']}: "
                  f"{m['compradores_no_topo']} compradores ({m['taxa_no_topo']:.1%}) | "
                  f"resultado R$ {m['resultado_rs']:,.0f}")
        a = bloco["aleatorio"]
        print(f"{'Aleatorio':22s} {a['taxa_no_topo']:.1%} | ~{a['compradores_esperados']:.0f} compradores | "
              f"resultado R$ {a['resultado_rs']:,.0f}")

    print("\n== Cenarios para 40 mil e-mails ==")
    for nome, c in resultado["cenarios_40mil_emails"].items():
        print(f"{nome:26s} taxa {c['taxa']:.1%} | {c['compras']:,} compras | resultado R$ {c['resultado_rs']:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nMetricas salvas em {out}")


if __name__ == "__main__":
    main()
