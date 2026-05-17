""" Configs do dataset de displasia """

from configs.datasets.base import DatasetConfig

DATASET_CONFIG = DatasetConfig(
    nome="displasia",
    tipo_tarefa="binario",
    nro_classes=1,
    # dir_dados = (...),
    # dir_treino = (...),
    # dir_val = (...),
    # dir_teste = (...)
)
