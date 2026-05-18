""" define as transformadas e, caso use, os aumentos de dados"""
import configs.datasets.base as base
from torchvision import transforms


class transformada_dados:
    """definir as transformadas a serem usadas nos dados de treino, validação e teste"""
    def __init__(self, config: base.DatasetConfig, usar_augmentation: bool):
        self.config = config
        self.usar_augmentation = usar_augmentation

    def obter_transformadas(self):
        """Método principal que retorna o par de transformadas correto baseado na configuração"""
        if self.usar_augmentation:
            return self.transformada_dados_aumentada()
        return self.transformadas_base()

    def transformadas_base(self):
        transformacao_treino = transforms.Compose([
            transforms.Resize(self.config.tam_input),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalizacao_mean, std=self.config.normalizacao_std),
        ])

        transformacao_teste = transforms.Compose([
            transforms.Resize(self.config.tam_input),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalizacao_mean, std=self.config.normalizacao_std),
        ])

        return transformacao_treino, transformacao_teste

    def transformada_dados_aumentada(self):
        """definir as transformadas a serem usadas nos dados de treino com aumento"""
        transformacao_treino = transforms.Compose([
            transforms.RandomResizedCrop(self.config.tam_input, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalizacao_mean, std=self.config.normalizacao_std),
        ])

        transformacao_teste = transforms.Compose([
            transforms.Resize(self.config.tam_input),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalizacao_mean, std=self.config.normalizacao_std),
        ])

        return transformacao_treino, transformacao_teste