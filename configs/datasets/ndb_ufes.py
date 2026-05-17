""" Configs do dataset NDB-UFES """

from configs.datasets.base import DatasetConfig

DATASET_CONFIG = DatasetConfig(
    nome="ndb",
    tipo_tarefa="multi",
    nro_classes=3,
    # dir_dados = (...),
    # dir_treino = (...),
    # dir_val = (...),
    # dir_teste = (...)
)