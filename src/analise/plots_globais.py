""" Gera os gráficos finais comparativos """

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
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
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "results" / "csv" / "consolidado.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "globais"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "tabelas" / "resumo_global.csv"
DEFAULT_TABLE_PDF_DIR = PROJECT_ROOT / "results" / "figures" / "tabelas_globais"


MODEL_PRETTY = {
    "resnet18": "ResNet18",
    "resnet-18": "ResNet18",
    "resnet_18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet-34": "ResNet34",
    "resnet_34": "ResNet34",
}

TRAINING_MODE_PRETTY = {
    "fs": "FS",
    "from_scratch": "FS",
    "from scratch": "FS",
    "pt_fc": "PT-FC",
    "pt-fc": "PT-FC",
    "pt fc": "PT-FC",
    "pt_all": "PT-ALL",
    "pt-all": "PT-ALL",
    "pt all": "PT-ALL",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
        ":": "",
        ".": "",
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
        "ç": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "O CSV consolidado está sem as colunas obrigatórias: "
            + ", ".join(sorted(missing))
        )


def standardize_model(value) -> str:
    key = str(value).strip().lower()
    return MODEL_PRETTY.get(key, str(value).strip())


def standardize_training_mode(value) -> str:
    key = str(value).strip().lower()
    return TRAINING_MODE_PRETTY.get(key, str(value).strip())


def standardize_augmentation(value) -> str:
    if isinstance(value, bool):
        return "com augmentation" if value else "sem augmentation"

    if pd.isna(value):
        return "sem augmentation"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "com augmentation" if bool(value) else "sem augmentation"

    key = str(value).strip().lower()
    truthy = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "sim",
        "com",
        "aug",
        "augmentation",
        "with",
        "with augmentation",
    }
    return "com augmentation" if key in truthy else "sem augmentation"


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_columns(df)

    df = df.copy()

    df["dataset_label"] = df["dataset"].astype(str).str.strip()
    df["model_label"] = df["model"].apply(standardize_model)
    df["training_mode_label"] = df["training_mode"].apply(standardize_training_mode)
    df["augmentation_label"] = df["augmentation"].apply(standardize_augmentation)

    numeric_cols = [
        "repetition",
        "seed",
        "acc_test",
        "f1_macro_test",
        "f1_weighted_test",
        "num_params",
        "gflops",
        "best_epoch",
        "val_acc_best",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["dataset_label", "model_label", "training_mode_label", "augmentation_label"],
            dropna=False,
        )
        .agg(
            repetitions=("repetition", "count"),
            acc_test_mean=("acc_test", "mean"),
            acc_test_std=("acc_test", "std"),
            f1_macro_test_mean=("f1_macro_test", "mean"),
            f1_macro_test_std=("f1_macro_test", "std"),
            f1_weighted_test_mean=("f1_weighted_test", "mean"),
            f1_weighted_test_std=("f1_weighted_test", "std"),
            num_params_mean=("num_params", "mean"),
            num_params_std=("num_params", "std"),
            gflops_mean=("gflops", "mean"),
            gflops_std=("gflops", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_std=("best_epoch", "std"),
            val_acc_best_mean=("val_acc_best", "mean"),
            val_acc_best_std=("val_acc_best", "std"),
        )
        .reset_index()
        .sort_values(
            by=["dataset_label", "model_label", "training_mode_label", "augmentation_label"]
        )
    )

    std_cols = [
        "acc_test_std",
        "f1_macro_test_std",
        "f1_weighted_test_std",
        "num_params_std",
        "gflops_std",
        "best_epoch_std",
        "val_acc_best_std",
    ]

    for col in std_cols:
        grouped[col] = grouped[col].fillna(0.0)

    return grouped


def format_mean_std(mean_value, std_value, decimals: int = 4) -> str:
    if pd.isna(mean_value):
        mean_value = 0.0
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def summary_table_for_pdf(summary: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    subset = summary[summary["dataset_label"] == dataset_name].copy()
    if subset.empty:
        return subset

    display = pd.DataFrame(
        {
            "Arquitetura": subset["model_label"],
            "Treino": subset["training_mode_label"],
            "Augmentation": subset["augmentation_label"],
            "Acc": [
                format_mean_std(m, s) for m, s in zip(subset["acc_test_mean"], subset["acc_test_std"])
            ],
            "F1 Macro": [
                format_mean_std(m, s)
                for m, s in zip(subset["f1_macro_test_mean"], subset["f1_macro_test_std"])
            ],
            "F1 Weighted": [
                format_mean_std(m, s)
                for m, s in zip(subset["f1_weighted_test_mean"], subset["f1_weighted_test_std"])
            ],
            "Parâmetros": [
                format_mean_std(m, s) for m, s in zip(subset["num_params_mean"], subset["num_params_std"])
            ],
            "GFLOPs": [
                format_mean_std(m, s) for m, s in zip(subset["gflops_mean"], subset["gflops_std"])
            ],
            "Best epoch": [
                format_mean_std(m, s) for m, s in zip(subset["best_epoch_mean"], subset["best_epoch_std"])
            ],
            "Val acc best": [
                format_mean_std(m, s)
                for m, s in zip(subset["val_acc_best_mean"], subset["val_acc_best_std"])
            ],
        }
    )

    return display


def save_table_pdf(display_df: pd.DataFrame, save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    if display_df.empty:
        return

    n_rows, n_cols = display_df.shape
    fig_width = max(12, n_cols * 1.7)
    fig_height = max(4, n_rows * 0.45 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=20)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(summary: pd.DataFrame, save_path: str | Path) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    summary.to_csv(save_path, index=False, encoding="utf-8")


def ordered_conditions(df: pd.DataFrame) -> list[tuple[str, str]]:
    available = {
        (str(m), str(a))
        for m, a in zip(df["training_mode_label"].astype(str), df["augmentation_label"].astype(str))
    }

    preferred = [
        ("FS", "sem augmentation"),
        ("FS", "com augmentation"),
        ("PT-FC", "sem augmentation"),
        ("PT-FC", "com augmentation"),
        ("PT-ALL", "sem augmentation"),
        ("PT-ALL", "com augmentation"),
    ]

    result = [item for item in preferred if item in available]
    if not result:
        result = sorted(available)
    return result


def model_order_from_df(df: pd.DataFrame) -> list[str]:
    preferred = ["ResNet18", "ResNet34"]
    available = df["model_label"].dropna().astype(str).unique().tolist()
    result = [m for m in preferred if m in available]
    if not result:
        result = sorted(available)
    return result


def plot_f1_global(
    summary: pd.DataFrame,
    dataset_name: str,
    metric_mean_col: str,
    metric_std_col: str,
    save_path: str | Path,
    y_label: str,
) -> None:
    ds = summary[summary["dataset_label"] == dataset_name].copy()
    if ds.empty:
        return

    model_order = model_order_from_df(ds)
    condition_order = ordered_conditions(ds)
    if not model_order or not condition_order:
        return

    x = np.arange(len(condition_order))
    width = 0.8 / max(len(model_order), 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, model_name in enumerate(model_order):
        model_df = ds[ds["model_label"] == model_name]
        means = []
        stds = []

        for training_mode, aug in condition_order:
            row = model_df[
                (model_df["training_mode_label"] == training_mode)
                & (model_df["augmentation_label"] == aug)
            ]

            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row.iloc[0][metric_mean_col]))
                stds.append(float(row.iloc[0][metric_std_col]))

        offset = (idx - (len(model_order) - 1) / 2.0) * width
        positions = x + offset

        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=model_name,
        )

    x_labels = [f"{mode}\n{aug}" for mode, aug in condition_order]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} global - {dataset_name}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Arquitetura")

    plt.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_architecture_metric(
    summary: pd.DataFrame,
    dataset_name: str,
    metric_mean_col: str,
    metric_std_col: str,
    save_path: str | Path,
    title: str,
    y_label: str,
) -> None:
    ds = summary[summary["dataset_label"] == dataset_name].copy()
    if ds.empty:
        return

    model_order = model_order_from_df(ds)
    if not model_order:
        return

    values = []
    errors = []

    for model_name in model_order:
        row = ds[ds["model_label"] == model_name]
        if row.empty:
            values.append(np.nan)
            errors.append(0.0)
        else:
            values.append(float(row.iloc[0][metric_mean_col]))
            errors.append(float(row.iloc[0][metric_std_col]))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_order))

    ax.bar(x, values, yerr=errors, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} - {dataset_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_reports(
    csv_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summary_csv: str | Path = DEFAULT_SUMMARY_CSV,
    table_pdf_dir: str | Path = DEFAULT_TABLE_PDF_DIR,
) -> pd.DataFrame:
    df = load_results(csv_path)
    summary = build_summary_table(df)

    output_dir = Path(output_dir)
    table_pdf_dir = Path(table_pdf_dir)

    ensure_dir(output_dir)
    ensure_dir(table_pdf_dir)

    save_summary_csv(summary, summary_csv)

    dataset_names = sorted(summary["dataset_label"].dropna().astype(str).unique().tolist())

    for dataset_name in dataset_names:
        dataset_slug = slugify(dataset_name)
        dataset_output_dir = output_dir / dataset_slug
        ensure_dir(dataset_output_dir)

        plot_f1_global(
            summary=summary,
            dataset_name=dataset_name,
            metric_mean_col="f1_macro_test_mean",
            metric_std_col="f1_macro_test_std",
            save_path=dataset_output_dir / "f1_macro_global.pdf",
            y_label="F1-score macro",
        )

        plot_f1_global(
            summary=summary,
            dataset_name=dataset_name,
            metric_mean_col="f1_weighted_test_mean",
            metric_std_col="f1_weighted_test_std",
            save_path=dataset_output_dir / "f1_weighted_global.pdf",
            y_label="F1-score weighted",
        )

        plot_architecture_metric(
            summary=summary,
            dataset_name=dataset_name,
            metric_mean_col="num_params_mean",
            metric_std_col="num_params_std",
            save_path=dataset_output_dir / "num_params_by_architecture.pdf",
            title="Número de parâmetros por arquitetura",
            y_label="Número de parâmetros",
        )

        plot_architecture_metric(
            summary=summary,
            dataset_name=dataset_name,
            metric_mean_col="gflops_mean",
            metric_std_col="gflops_std",
            save_path=dataset_output_dir / "gflops_by_architecture.pdf",
            title="GFLOPs por arquitetura",
            y_label="GFLOPs",
        )

        table_df = summary_table_for_pdf(summary, dataset_name)
        save_table_pdf(
            display_df=table_df,
            save_path=table_pdf_dir / f"resumo_{dataset_slug}.pdf",
            title=f"Resumo global - {dataset_name}",
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera os gráficos globais e a tabela resumo a partir do CSV consolidado."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help="Caminho do CSV consolidado com todas as execuções.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório onde os gráficos serão salvos.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help="Caminho do CSV resumo com média e desvio padrão.",
    )
    parser.add_argument(
        "--table_pdf_dir",
        type=str,
        default=str(DEFAULT_TABLE_PDF_DIR),
        help="Diretório onde as tabelas em PDF serão salvas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_reports(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        table_pdf_dir=args.table_pdf_dir,
    )


if __name__ == "__main__":
    main()