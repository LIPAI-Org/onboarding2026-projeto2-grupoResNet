""" Carregar imagens e definir labels baseando-se nos arquivos CSV do professor """
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import src.data.transformadas as transformadas
import configs.datasets.base as base


class HistologiaDatasetCustom(Dataset):
    """Classe auxiliar para ler imagens e labels a partir de um arquivo CSV específico."""
    def __init__(self, root_dir, csv_path, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        try:
            df = pd.read_csv(csv_path)
            if 'split' not in df.columns and 'sets' not in df.columns and ';' in ''.join(df.columns):
                df = pd.read_csv(csv_path, sep=';')
        except Exception as e:
            raise FileNotFoundError(
                f"Erro ao carregar o arquivo CSV em '{csv_path}'. "
                f"Certifique-se de que o caminho está correto. Detalhes: {e}"
            )
            
        self.col_split = 'split' if 'split' in df.columns else ('sets' if 'sets' in df.columns else None)
        self.col_path = 'caminho_imagem' if 'caminho_imagem' in df.columns else ('path' if 'path' in df.columns else None)
        self.col_label = 'label' if 'label' in df.columns else ('label_number' if 'label_number' in df.columns else None)
        
        if not self.col_split or not self.col_path or not self.col_label:
            raise KeyError(
                f"\n❌ ERRO DE FORMATAÇÃO NO CSV: Não consegui mapear as colunas em '{csv_path}'.\n"
                f"Colunas encontradas: {list(df.columns)}.\n"
                f"Espera-se colunas de split (split ou sets), caminho (caminho_imagem ou path) e rótulo (label ou label_number)."
            )
        
        self.data = df[df[self.col_split] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_relative_path = self.data.iloc[idx][self.col_path]
        label = int(self.data.iloc[idx][self.col_label])
        
        img_path = os.path.join(self.root_dir, img_relative_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class datasets:
    def __init__(self, config: base.DatasetConfig, escolha_transformada: str):
        self.config = config
        self.escolha_transformada = escolha_transformada
        
        self.root_path_ROI = 'data/raw/Original_ROI_images'
        self.root_path_NDB = 'data/raw/images_NDB-UFES'
        self.csv_ROI = 'data/splits/manifest_split_oralepitheliumdb.csv'
        self.csv_NDB = 'data/splits/manifest_split_multiclass_NDB-UFES.csv'

        gerador_base = transformadas.transformada_dados(config=self.config, usar_augmentation=False)
        self.treino_base, self.teste_base = gerador_base.obter_transformadas()
        
        gerador_aumentado = transformadas.transformada_dados(config=self.config, usar_augmentation=True)
        self.treino_aumentado, _ = gerador_aumentado.obter_transformadas()

    def carregar_dados_ROI_base(self):
        """carregar os dados do dataset ROI sem aumento (Treino, Validação e Teste)"""
        train_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='train', transform=self.treino_base)
        val_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='val', transform=self.teste_base) 
        test_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='test', transform=self.teste_base) 
        return train_dataset, val_dataset, test_dataset

    def carregar_dados_NDB_base(self):
        """carregar os dados do dataset NDB-UFES sem aumento (Treino, Validação e Teste)"""
        train_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='train', transform=self.treino_base)
        val_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='val', transform=self.teste_base)
        test_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='test', transform=self.teste_base)
        return train_dataset, val_dataset, test_dataset

    def carregar_dados_ROI_aumentado(self):
        """carregar os dados do dataset ROI com aumento no treino"""
        train_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='train', transform=self.treino_aumentado)
        val_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='val', transform=self.teste_base)
        test_dataset = HistologiaDatasetCustom(self.root_path_ROI, self.csv_ROI, split='test', transform=self.teste_base)
        return train_dataset, val_dataset, test_dataset

    def carregar_dados_NDB_aumentado(self):
        """carregar os dados do dataset NDB-UFES com aumento no treino"""
        train_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='train', transform=self.treino_aumentado)
        val_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='val', transform=self.teste_base)
        test_dataset = HistologiaDatasetCustom(self.root_path_NDB, self.csv_NDB, split='test', transform=self.teste_base)
        return train_dataset, val_dataset, test_dataset