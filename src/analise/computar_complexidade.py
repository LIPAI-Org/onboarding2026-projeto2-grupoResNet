""" Calcula o nro de parâmetros e os GFLOPS """

import os
import csv

from src.treino.custo import (
    count_parameters,
    count_trainable_parameters,
    calcular_gflops
)
from configs.datasets.base import DatasetConfig


def compute_model_complexity(
    model,
    model_name,
    config_dataset: DatasetConfig
):

    total_params = count_parameters(
        model
    )

    trainable_params = (
        count_trainable_parameters(
            model
        )
    )

    gflops = calcular_gflops(
        model,
        config_dataset
    )

    results = {

        "model": model_name,

        "total_params":
            total_params,

        "trainable_params":
            trainable_params,

        "gflops":
            gflops
    }

    return results


def save_complexity_results(
    results,
    save_path
):

    save_dir = os.path.dirname(
        save_path
    )

    if save_dir != "":

        os.makedirs(
            save_dir,
            exist_ok=True
        )

    file_exists = os.path.exists(
        save_path
    )

    with open(
    save_path,
    mode="a",
    newline="",
    encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        if not file_exists:

            writer.writerow(
                [
                    "model",
                    "total_params",
                    "trainable_params",
                    "gflops"
                ]
            )

        writer.writerow(
            [
                results["model"],
                results["total_params"],
                results["trainable_params"],
                results["gflops"]
            ]
        )