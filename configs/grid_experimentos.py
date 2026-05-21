""" Define as possibilidades de experimentos """

from itertools import product
from dataclasses import asdict

from configs.datasets.displasia import DATASET_CONFIG as DISPLASIA_CONFIG
from configs.datasets.ndb_ufes import DATASET_CONFIG as NDB_CONFIG

SEEDS = [42, 123, 2025]

MODELOS = [
    "resnet18",
    "resnet34"
]

MODOS_TREINAMENTO = [
    "fs",
    "pt_fc",
    "pt_all"
]

AUMENTO = [
    True,
    False
]

DATASETS = {
    "displasia": DISPLASIA_CONFIG,
    "ndb": NDB_CONFIG
}

def construir_grid_experimentos():
    """ Gera a lista completa de experimentos """
    grid = []

    for seed, nome_modelo, nome_dataset, modo_treino, usa_aug in product(
        SEEDS,
        MODELOS,
        DATASETS.keys(),
        MODOS_TREINAMENTO,
        AUMENTO
    ):
        dataset_cfg = DATASETS[nome_dataset]

        experimento = {
            "seed": seed,
            "modelo": nome_modelo,
            "dataset": nome_dataset,
            "dataset_config": asdict(dataset_cfg),
            "modo_treinamento": modo_treino,
            "aumento": usa_aug
        }
        grid.append(experimento)
    
    return grid

GRID_EXPERIMENTOS = construir_grid_experimentos()
