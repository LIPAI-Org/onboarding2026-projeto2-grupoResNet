""" Centraliza a lógica do DataLoader() """

import configs.datasets.base as base
from torch.utils.data import DataLoader
import configs.configs_base as cb

class DataLoaders:
    def __init__(self, config: base.DatasetConfig, datasets, batch_size=cb.BATCH_SIZE):
        self.config = config
        self.datasets = datasets
        self.batch_size = batch_size

    def _obter_datasets_por_nome(self, cenario=None):
        """Método auxiliar para mapear qual função da sua classe datasets deve ser chamada."""
        nome_dataset = self.config.nome.lower()
        
        if cenario is None:
            cenario = self.datasets.escolha_transformada
        
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
        """criar os dataloaders para treino, validação e teste sem aumento"""
        train_dataset, val_dataset, test_dataset = self._obter_datasets_por_nome(cenario="base")
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader
    
    def criar_dataloaders_aumentados(self):
        """criar os dataloaders para treino, validação e teste com aumento"""
        train_dataset_aumentado, val_dataset_aumentado, test_dataset_aumentado = self._obter_datasets_por_nome(cenario="aumentado")
        
        train_loader = DataLoader(
            dataset=train_dataset_aumentado, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset=val_dataset_aumentado, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset_aumentado, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader