""" Centraliza a lógica do DataLoader() """

import configs.datasets.base as base
import torch
from torch.utils.data import DataLoader

class DataLoaders:
    def __init__(self, config: base.DatasetConfig, datasets):
        self.config = config
        self.datasets = datasets
        
        # Define um batch_size padrão caso ele não esteja mapeado na sua config ainda
        self.batch_size = getattr(self.config, 'batch_size', 32)

    def _obter_datasets_por_nome(self, cenario: str):
        """Método auxiliar para mapear qual função da sua classe datasets deve ser chamada."""
        nome_dataset = self.config.nome.lower()
        
        if cenario == "base":
            if "roi" in nome_dataset or "displasia" in nome_dataset:
                return self.datasets.carregar_dados_ROI_base()
            else:
                return self.datasets.carregar_dados_NDB_base()
                
        elif cenario == "aumentado":
            if "roi" in nome_dataset or "displasia" in nome_dataset:
                return self.datasets.carregar_dados_ROI_aumentado()
            else:
                return self.datasets.carregar_dados_NDB_aumentado()
        
        raise ValueError(f"Cenário desconhecido: {cenario}")

    def criar_dataloaders_base(self):
        """criar os dataloaders para treino e teste sem aumento"""
        # Chama a função correta da sua classe datasets para pegar os objetos Dataset
        train_dataset, test_dataset = self._obter_datasets_por_nome(cenario="base")
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return train_loader, test_loader
    
    def criar_dataloaders_aumentados(self):
        """criar os dataloaders para treino e teste com aumento"""
        # Chama a função correta da sua classe datasets para pegar os objetos Dataset
        train_dataset_aumentado, test_dataset_aumentado = self._obter_datasets_por_nome(cenario="aumentado")
        
        train_loader = DataLoader(
            dataset=train_dataset_aumentado, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset_aumentado, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return train_loader, test_loader