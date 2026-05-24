""" Visualiza os experiments"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "planilha_resultados.csv"
)


DISPLAY_COLUMNS = [
    "repetition",
    "seed",
    "dataset",
    "model",
    "training_mode",
    "augmentation",
    "acc_test",
    "f1_macro_test",
    "f1_weighted_test",
    "num_params",
    "gflops",
    "best_epoch",
    "val_acc_best",
]


def carregar_resultados(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    for col in DISPLAY_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Coluna obrigatória ausente: {col}"
            )

    return df


def aplicar_filtros(
    df: pd.DataFrame,
    dataset=None,
    model=None,
    training_mode=None,
    augmentation=None,
    seed=None,
    repetition=None,
):
    resultado = df.copy()

    if dataset is not None:
        resultado = resultado[
            resultado["dataset"].astype(str).str.lower()
            == str(dataset).lower()
        ]

    if model is not None:
        resultado = resultado[
            resultado["model"].astype(str).str.lower()
            == str(model).lower()
        ]

    if training_mode is not None:
        resultado = resultado[
            resultado["training_mode"].astype(str).str.lower()
            == str(training_mode).lower()
        ]

    if augmentation is not None:
        aug = str(augmentation).lower()

        if aug in {"true", "1", "yes", "sim"}:
            aug = True
        elif aug in {"false", "0", "no", "nao"}:
            aug = False

        resultado = resultado[
            resultado["augmentation"] == aug
        ]

    if seed is not None:
        resultado = resultado[
            resultado["seed"] == seed
        ]

    if repetition is not None:
        resultado = resultado[
            resultado["repetition"] == repetition
        ]

    return resultado


def ordenar_resultados(
    df: pd.DataFrame,
    metric: str = "f1_macro_test",
    ascending: bool = False,
):
    if metric not in df.columns:
        raise ValueError(
            f"Métrica inválida: {metric}"
        )

    return df.sort_values(
        by=metric,
        ascending=ascending,
    )


def mostrar_resultados(
    df: pd.DataFrame,
    limit=None,
):
    if df.empty:
        print("\nNenhum resultado encontrado.\n")
        return

    resultado = df.copy()

    resultado = resultado[DISPLAY_COLUMNS]

    resultado = resultado.round(4)

    if limit is not None:
        resultado = resultado.head(limit)

    print()
    print(resultado.to_string(index=False))
    print()
    print(f"Total de resultados: {len(resultado)}")
    print()


def mostrar_melhor_resultado(
    df: pd.DataFrame,
    metric: str = "f1_macro_test",
):
    if df.empty:
        print("\nNenhum resultado encontrado.\n")
        return

    melhor = ordenar_resultados(
        df,
        metric=metric,
        ascending=False,
    ).head(1)

    print()
    print("MELHOR RESULTADO")
    print()

    mostrar_resultados(melhor)


def mostrar_resumo(
    df: pd.DataFrame,
):
    if df.empty:
        print("\nNenhum resultado encontrado.\n")
        return

    resumo = (
        df.groupby(
            [
                "dataset",
                "model",
                "training_mode",
                "augmentation",
            ]
        )
        .agg(
            acc_test_mean=("acc_test", "mean"),
            acc_test_std=("acc_test", "std"),
            f1_macro_mean=("f1_macro_test", "mean"),
            f1_macro_std=("f1_macro_test", "std"),
            f1_weighted_mean=("f1_weighted_test", "mean"),
            f1_weighted_std=("f1_weighted_test", "std"),
        )
        .reset_index()
    )

    resumo = resumo.round(4)

    print()
    print(resumo.to_string(index=False))
    print()


def mostrar_top_resultados(
    df: pd.DataFrame,
    top_n: int = 10,
    metric: str = "f1_macro_test",
):
    if df.empty:
        print("\nNenhum resultado encontrado.\n")
        return

    top_df = ordenar_resultados(
        df,
        metric=metric,
        ascending=False,
    ).head(top_n)

    print()
    print(f"TOP {top_n} RESULTADOS")
    print()

    mostrar_resultados(top_df)


def listar_valores_unicos(df: pd.DataFrame):
    print()
    print("DATASETS:")
    print(sorted(df["dataset"].astype(str).unique().tolist()))
    print()

    print("MODELOS:")
    print(sorted(df["model"].astype(str).unique().tolist()))
    print()

    print("TRAINING MODES:")
    print(sorted(df["training_mode"].astype(str).unique().tolist()))
    print()

    print("AUGMENTATIONS:")
    print(sorted(df["augmentation"].astype(str).unique().tolist()))
    print()

    print("SEEDS:")
    print(sorted(df["seed"].unique().tolist()))
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualização e filtragem dos experimentos."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
    )

    parser.add_argument(
        "--dataset",
        type=str,
    )

    parser.add_argument(
        "--model",
        type=str,
    )

    parser.add_argument(
        "--training_mode",
        type=str,
    )

    parser.add_argument(
        "--augmentation",
        type=str,
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--repetition",
        type=int,
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="f1_macro_test",
    )

    parser.add_argument(
        "--ascending",
        action="store_true",
    )

    parser.add_argument(
        "--top",
        type=int,
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--best",
        action="store_true",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
    )

    parser.add_argument(
        "--list_values",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = carregar_resultados(args.csv_path)

    filtrado = aplicar_filtros(
        df=df,
        dataset=args.dataset,
        model=args.model,
        training_mode=args.training_mode,
        augmentation=args.augmentation,
        seed=args.seed,
        repetition=args.repetition,
    )

    if args.list_values:
        listar_valores_unicos(filtrado)
        return

    if args.summary:
        mostrar_resumo(filtrado)
        return

    if args.best:
        mostrar_melhor_resultado(
            filtrado,
            metric=args.metric,
        )
        return

    if args.top is not None:
        mostrar_top_resultados(
            filtrado,
            top_n=args.top,
            metric=args.metric,
        )
        return

    filtrado = ordenar_resultados(
        filtrado,
        metric=args.metric,
        ascending=args.ascending,
    )

    mostrar_resultados(
        filtrado,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()