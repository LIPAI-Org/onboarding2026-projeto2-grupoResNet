""" Gera os pdfs vetoriais da matriz de confusão """

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
    normalize=False,
    figsize=(8, 8)
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    if normalize:

        cm = cm.astype("float")

        cm = cm / cm.sum(
            axis=1,
            keepdims=True
        )

        cm = np.nan_to_num(cm)

    _, ax = plt.subplots(
    figsize=figsize
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        colorbar=True
    )

    ax.set_xlabel(
        "predicted label"
    )

    ax.set_ylabel(
        "true label"
    )

    if normalize:

        ax.set_title(
            "normalized confusion matrix"
        )

    else:

        ax.set_title(
            "confusion matrix"
        )

    save_dir = os.path.dirname(
        save_path
    )

    if save_dir != "":

        os.makedirs(
            save_dir,
            exist_ok=True
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()


def save_confusion_matrix_data(
    y_true,
    y_pred,
    save_path
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    save_dir = os.path.dirname(
        save_path
    )

    if save_dir != "":

        os.makedirs(
            save_dir,
            exist_ok=True
        )

    np.save(
        save_path,
        cm
    )