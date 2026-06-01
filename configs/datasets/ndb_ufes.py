""" Configs do dataset NDB-UFES """

from configs.datasets.base import DatasetConfig

DATASET_CONFIG = DatasetConfig(
    nome="ndb",
    tipo_tarefa="multi",
    nro_classes=3,
    labels_classes=["LO c/ displasia", "LO s/ displasia", "OSCC"]
)