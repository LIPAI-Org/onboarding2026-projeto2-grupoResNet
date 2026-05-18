""" Carregar imagens e definir labels baseando-se nos arquivos CSV do professor """
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import src.data.transformadas as transformadas


class HistologiaDatasetCustom(Dataset):
    """Classe auxiliar para ler imagens e labels a partir de um arquivo CSV específico."""
    def __init__(self, root_dir, csv_path, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_relative_path = self.data.iloc[idx]['caminho_imagem']
        label = int(self.data.iloc[idx]['label'])
        
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class datasets:
    def __init__(self, config):
        self.config = config
        
        self.root_path_ROI = 'data/raw/Original_ROI_images'
        self.root_path_NDB = 'data/raw/images_NDB-UFES'
        
        self.csv_ROI = 'data/splits/manifest_split_oralepitheliumdb.csv'
        self.csv_NDB = 'data/splits/manifest_split_multiclass_NDB-UFES.csv'
        
        gerador_transformadas = transformadas.transformada_dados(config=self.config, usar_augmentation=False)
        self.treino_base, self.teste_base = gerador_transformadas.transformadas_base()
        
        gerador_aumentado = transformadas.transformada_dados(config=self.config, usar_augmentation=True)
        self.treino_aumentado, self.teste_aumentado = gerador_aumentado.transformada_dados_aumentada()

    def carregar_dados_ROI_base(self):
        """carregar os dados do dataset ROI sem aumento (Treino e Validação/Teste)"""
        train_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='train', transform=self.treino_base)
        test_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='val', transform=self.teste_base) 
        return train_dataset, test_dataset

    def carregar_dados_NDB_base(self):
        """carregar os dados do dataset NDB-UFES sem aumento"""
        train_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='train', transform=self.treino_base)
        test_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='val', transform=self.teste_base)
        return train_dataset, test_dataset

    def carregar_dados_ROI_aumentado(self):
        """carregar os dados do dataset ROI com aumento no treino"""
        train_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='train', transform=self.treino_aumentado)
        test_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='val', transform=self.teste_aumentado)
        return train_dataset, test_dataset

    def carregar_dados_NDB_aumentado(self):
        """carregar os dados do dataset NDB-UFES com aumento no treino"""
        train_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='train', transform=self.treino_aumentado)
        test_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='val', transform=self.teste_aumentado)
        return train_dataset, test_dataset