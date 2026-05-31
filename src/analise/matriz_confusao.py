""" Gera os pdfs vetoriais da matriz de confusão """

import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def salvar_matriz_confusao(cm, path_saida, nome_arquivo="matriz_confusao.png", classes=None):
    """
    Salva o plot de uma matriz de confusão em disco.

    Parâmetros:
        cm: matriz de confusão já calculada com confusion_matrix(y_true, y_pred)
        path_saida: diretório onde o arquivo será salvo
        nome_arquivo: nome do arquivo de saída
        classes: lista com os nomes das classes (opcional)
    """
    os.makedirs(path_saida, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    plt.tight_layout()
    caminho_completo = os.path.join(path_saida, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300, bbox_inches="tight")
    plt.close(fig)