""" Funções para rodar os experimentos """

from itertools import product
from dataclasses import asdict

from configs.configs_base import DEVICE
from src.data.datasets import datasets
from src.data.dataloaders import DataLoaders
from configs.datasets.base import DatasetConfig
from src.utils.seed import definir_seed
from src.modelos.modelo_factory import get_model
from src.analise.computar_complexidade import compute_model_complexity
from src.treino.treinador import train_model
from src.treino.avaliador import evaluate_model
from src.utils.escritor_csv import escrever_resultados_csv
from src.analise.matriz_confusao import salvar_matriz_confusao
import configs.grid_experimentos as gdexp
import src.analise.curvas_aprendizado as cva
import src.utils.paths as paths

def rodar_experimento(
        experimento: dict,
        num_teste: int,
        total_testes: int
):
    """
    Roda um experimento específico, com experimento sendo definido como:\n
    experimento = {
        "seed": int,
        "modelo": str,
        "dataset": str,
        "dataset_config": asdict(DatasetConfig),
        "modo_treinamento": str,
        "aumento": bool
    }
    para os valores possíveis de cada entrada,
    referir-se ao arquivo configs/grid_experimentos.py
    """
    seed = experimento["seed"]
    definir_seed(seed)

    config_dataset = DatasetConfig(**experimento["dataset_config"])

    modo_treinamento = experimento["modo_treinamento"]

    modelo = get_model(experimento["modelo"], config_dataset.nro_classes, modo_treinamento)
    
    print("\n" + "="*70)
    if num_teste and total_testes:
        print(f"[TESTE {num_teste}/{total_testes}] Configurando Ambiente...")
    else:
        print("Configurando Ambiente do Experimento...")
    print(f"• Modelo: {experimento.get('modelo')} | Modo: {modo_treinamento}")
    print(f"• Dataset: {experimento.get('dataset')} | Data Augmentation: {experimento.get('aumento')}")
    print(f"• Seed: {seed}")
    print("="*70)

    
    device = DEVICE
    modelo = modelo.to(device)

    complexidade = compute_model_complexity(modelo,
                                            experimento["modelo"],
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
    
    print("\n[Treino] Iniciando a execução das épocas...")
    resultados_treino = train_model(
        modelo,
        train_loader,
        val_loader,
        config_dataset,
        seed
    )

    print("[Avaliação] Computando predições no conjunto de teste...")
    resultados_teste = evaluate_model(
        modelo,
        test_loader,
        config_dataset,
        seed
    )
    
    print("[Salvar] Registrando resultados no arquivo CSV...")
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

    print("[Salvar] Salvando o plot de custo e acuracia...")
    cva.plotar_e_salvar_loss(
        str(seed),
        str(experimento.get("modelo")),
        str(modo_treinamento),
        str(aumento),
        str(experimento.get("dataset")),
        resultados_treino.get("history")
    )
    cva.plotar_e_salvar_acc(
        str(seed),
        str(experimento.get("modelo")),
        str(modo_treinamento),
        str(aumento),
        str(experimento.get("dataset")),
        resultados_treino.get("history")
    )
    salvar_matriz_confusao(
        cm=resultados_teste.get("confusion_matrix"),
        path_saida=paths.PATH_MATRIZ,
        nome_arquivo=f'{str(seed)}_{str(experimento.get("modelo"))}_{str(modo_treinamento)}_{str(experimento.get("dataset"))}',
        classes=config_dataset.labels_classes
    )
    print("Concluído com sucesso!\n")

def rodar_experimentos_baseado_em_parametros(
        seed: int | None = None,
        modelo: str | None = None,
        modo_treinamento: str | None = None,
        aumento: bool | None = None,
        dataset: str | None = None,
):
    """
    Roda os experimentos a partir de valores fixos
    possivelmente passados como parâmetros.
    """
    seeds = [seed] if seed is not None else gdexp.SEEDS
    modelos = [modelo] if modelo is not None else gdexp.MODELOS
    modos_treinamento = [modo_treinamento] if modo_treinamento is not None else gdexp.MODOS_TREINAMENTO
    aumentos = [aumento] if aumento is not None else gdexp.AUMENTO
    datasetes = [dataset] if dataset is not None else list(gdexp.DATASETS.keys())

    total_filtrados = len(seeds) * len(modelos) * len(datasetes) * len(modos_treinamento) * len(aumentos)
    idx = 1

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
        rodar_experimento(experimento, num_teste=idx, total_testes=total_filtrados)
        idx += 1

def rodar_todos_experimentos():
    """
    Roda todo experimento possível.
    """
    lista_experimentos = gdexp.GRID_EXPERIMENTOS
    total_testes = len(lista_experimentos)
    
    for idx, experimento in enumerate(lista_experimentos, start=1):
        rodar_experimento(experimento, num_teste=idx, total_testes=total_testes)