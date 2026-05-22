""" Responsável pelos calculos de custo durante o treinamento """

import torch

from thop import profile

from configs.datasets.base import DatasetConfig
from configs.configs_base import DEVICE

def count_parameters(model):
    return sum(
        p.numel() 
        for p in model.parameters()
    )


def count_trainable_parameters(model):
    return sum(
        p.numel() 
        for p in model.parameters() 
        if p.requires_grad
    )

def calcular_gflops(model, config_dataset: DatasetConfig):
    """
    Calcula GFLOPs
    """

    tam_input = (1,
                 config_dataset.canais_input,
                 config_dataset.tam_input[0],
                 config_dataset.tam_input[1]
                 )
    
    tensor_input = torch.randn(tam_input).to(DEVICE)

    profile_result = profile(
        model,
        inputs=(tensor_input,),
        verbose=False
    )

    macs = profile_result[0]

    gflops = (2 * macs) / 1e9

    return round(gflops, 4)
