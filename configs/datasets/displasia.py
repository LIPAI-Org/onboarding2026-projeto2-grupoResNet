""" Configs do dataset de displasia """

from configs.datasets.base import DatasetConfig

DATASET_CONFIG = DatasetConfig(
    nome="displasia",
    tipo_tarefa="binario",
    nro_classes=1,
    labels_classes= ["Healthy", "Severe"]
)
