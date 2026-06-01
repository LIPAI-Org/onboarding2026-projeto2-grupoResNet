""" Plota custo e acurácia de treino e validação """

import matplotlib.pyplot as plt
import os
import src.utils.paths as paths
from configs.configs_base import NUM_EPOCAS

def plotar_e_salvar_loss(
        seed: str,
        nome_modelo: str,
        modo_treinamento: str,
        aumento: str,
        dataset: str,
        history: dict[str, list],
    ):
    

    if not os.path.isdir(paths.PATH_CURVAS_APRENDIZADO):
        os.makedirs(
            paths.PATH_CURVAS_APRENDIZADO,
            exist_ok=True
        )

    nome_arq = (
        f"loss_{seed}_{nome_modelo}_"
        f"{modo_treinamento}_{aumento}_{dataset}.pdf"
    )

    caminho_saida = os.path.join(
        paths.PATH_CURVAS_APRENDIZADO,
        nome_arq
    )

    epocas = range(1, NUM_EPOCAS + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epocas, history["train_loss"], label="Train Loss")
    plt.plot(epocas, history["val_loss"], label="Val Loss")

    plt.title("Loss de Treino e Validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        caminho_saida,
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plotar_e_salvar_acc(
        seed: str,
        nome_modelo: str,
        modo_treinamento: str,
        aumento: str,
        dataset: str,
        history: dict[str, list],
    ):

    if not os.path.isdir(paths.PATH_CURVAS_APRENDIZADO):
        os.makedirs(
            paths.PATH_CURVAS_APRENDIZADO,
            exist_ok=True
        )

    nome_arq = (
        f"acc_{seed}_{nome_modelo}_"
        f"{modo_treinamento}_{aumento}_{dataset}.pdf"
    )

    caminho_saida = os.path.join(
        paths.PATH_CURVAS_APRENDIZADO,
        nome_arq
    )

    epocas = range(1, NUM_EPOCAS + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epocas,
        history["train_acc"],
        label="Acuracia Treino"
    )

    plt.plot(
        epocas,
        history["val_acc"],
        label="Acuracia Validacao"
    )

    plt.title("Acuracia de Treino e Validação")
    plt.xlabel("Época")
    plt.ylabel("Acc")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        caminho_saida,
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()