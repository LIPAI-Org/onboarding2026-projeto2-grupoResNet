""" Escrever os resultados na planilha final """

import csv
import os
from src.utils.paths import PATH_PLANILHA_RESULTADOS

__COLUNAS = [
    "repetition", "seed", "dataset", "model", "training_mode", "augmentation",
    "acc_test", "f1_macro_test", "f1_weighted_test", "num_params",
    "gflops", "best_epoch", "val_acc_best"
]

__CHAVES_GRUPO = ("seed", "dataset", "model", "training_mode", "augmentation")


def escrever_resultados_csv(
    seed: str, dataset: str, model: str, training_mode: str,
    augmentation: str, acc_test: str, f1_macro_test : str,
    f1_weighted_test: str, num_params: str, gflops: str,
    best_epoch: str, val_acc_best: str
) -> None:
    """
    Escreve no csv de resultados, dado os argumentos.
    TODOS OS ARGS DEVEM SER STRINGS!!! Lidar com isto!
    """
    nova_base = {
        "seed": seed,
        "dataset": dataset,
        "model": model,
        "training_mode": training_mode,
        "augmentation": augmentation,
    }

    repetition = 1

    arquivo_existe = os.path.exists(PATH_PLANILHA_RESULTADOS)
    if arquivo_existe:
        with open(PATH_PLANILHA_RESULTADOS, mode="r", newline="", encoding="utf-8") as arquivo:
            leitor = csv.DictReader(arquivo)

            for linha in leitor:
                mesma_combinacao = all(linha[chave] == nova_base[chave] for chave in __CHAVES_GRUPO)

                if mesma_combinacao:
                    try:
                        rep_atual = int(linha["repetition"])
                        if rep_atual >= repetition:
                            repetition = rep_atual + 1
                    except (ValueError, KeyError):
                        pass

    nova_linha = {
        "repetition": str(repetition),
        "seed": seed,
        "dataset": dataset,
        "model": model,
        "training_mode": training_mode,
        "augmentation": augmentation,
        "acc_test": acc_test,
        "f1_macro_test": f1_macro_test,
        "f1_weighted_test": f1_weighted_test,
        "num_params": num_params,
        "gflops": gflops,
        "best_epoch": best_epoch,
        "val_acc_best": val_acc_best,
    }

    with open(PATH_PLANILHA_RESULTADOS, mode="a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=__COLUNAS)

        if not arquivo_existe:
            escritor.writeheader()

        escritor.writerow(nova_linha)
