""" Configs bases dos datasets """
from dataclasses import dataclass
from typing import Literal

TipoTarefa = Literal["binario", "multi"]

@dataclass(frozen=True)
class DatasetConfig:
    tipo_tarefa: TipoTarefa
    nro_classes: int
    nome: str 
    tam_input: tuple[int, int] = (224, 224)
    canais_input: int = 3   # RGB

    # caminhos
    dir_dados: str = ""
    dir_treino: str = ""
    dir_val: str = ""
    dir_teste: str = ""

    # pré-processamento
    normalizacao_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalizacao_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def is_binario(self):
        return self.tipo_tarefa == "binario"
    
    @property
    def is_multi(self):
        return self.tipo_tarefa == "multi"