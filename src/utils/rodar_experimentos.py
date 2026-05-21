""" Funções para rodar os experimentos """

from itertools import product
from dataclasses import asdict

from src.data.datasets import datasets
from src.data.dataloaders import DataLoaders
from configs.datasets.base import DatasetConfig
from src.utils.seed import definir_seed
from src.modelos.modelo_factory import get_model
from src.analise.computar_complexidade import compute_model_complexity
from src.treino.treinador import train_model
from treino.avaliador import evaluate_model
from src.utils.escritor_csv import escrever_resultados_csv
import configs.grid_experimentos as gdexp

def rodar_experimento(
        experimento: dict
) -> None:
    seed = experimento.get("seed")
    definir_seed(seed)

    config_dataset = DatasetConfig(**experimento.get("dataset_config"))

    modo_treinamento = experimento.get("modo_treinamento")

    modelo = get_model(experimento.get("modelo"), config_dataset.nro_classes, modo_treinamento)

    complexidade = compute_model_complexity(modelo,
                                            experimento.get("modelo"),
                                            config_dataset
    )

    aumento = experimento.get("aumento")

    if not aumento:
        dataset = datasets(config=config_dataset, escolha_transformada="base")
    else:
        dataset = datasets(config=config_dataset, escolha_transformada="aumentada")
    
    dataloaders = DataLoaders(config_dataset, dataset)
    if not aumento:
        train_loader, val_loader, test_loader = dataloaders.criar_dataloaders_base()
    else:
        train_loader, val_loader, test_loader = dataloaders.criar_dataloaders_aumentados()
    
    resultados_treino = train_model(
        modelo,
        train_loader,
        val_loader,
        config_dataset,
        seed
    )

    resultados_teste = evaluate_model(
        modelo,
        test_loader,
        config_dataset,
        seed
    )
    
    escrever_resultados_csv(
        str(seed),
        str(experimento.get("dataset")),
        str(experimento.get("modelo")),
        str(modo_treinamento),
        str(aumento),
        str(resultados_teste.get("acc")),
        str(resultados_teste.get("f1_macro")),
        str(resultados_teste.get("f1_weighted")),
        str(complexidade.get("total_params")),
        str(complexidade.get("gflops")),
        str(resultados_treino.get("best_epoch")),
        str(resultados_treino.get("best_val_acc"))
    )

def rodar_experimentos_baseado_em_parametros(
        seed: int | None = None,
        modelo: str | None = None,
        modo_treinamento: str | None = None,
        aumento: bool | None = None,
        dataset: str | None = None,
):
    seeds = [seed] if seed is not None else gdexp.SEEDS
    modelos = [modelo] if modelo is not None else gdexp.MODELOS
    modos_treinamento = [modo_treinamento] if modo_treinamento is not None else gdexp.MODOS_TREINAMENTO
    aumentos = [aumento] if aumento is not None else gdexp.AUMENTO
    datasetes = [dataset] if dataset is not None else list(gdexp.DATASETS.keys())

    for seed_i, modelo_i, dataset_i, modo_i, aumento_i in product(
        seeds,
        modelos,
        datasetes,
        modos_treinamento,
        aumentos
    ):
        config_dataset = gdexp.DATASETS[dataset_i]
        experimento = {
            "seed": seed_i,
            "modelo": modelo_i,
            "dataset": dataset_i,
            "dataset_config": asdict(config_dataset),
            "modo_treinamento": modo_i,
            "aumento": aumento_i
        }
        rodar_experimento(experimento)

def rodar_todos_experimentos():
    for experimento in gdexp.GRID_EXPERIMENTOS:
        rodar_experimento(experimento)