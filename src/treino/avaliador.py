""" Responsável pelo calculo das métricas de treinamento (acurácia, etc)"""

import numpy as np
import torch

from tqdm import tqdm

from .metricas import calculate_metrics


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    checkpoint_path=None
):

    if checkpoint_path is not None:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    model = model.to(device)

    model.eval()

    test_losses = []

    all_preds = []
    all_targets = []

    with torch.no_grad():

        test_bar = tqdm(
            test_loader,
            desc="testing"
        )

        for images, labels in test_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            test_losses.append(
                loss.item()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

    test_loss = np.mean(
        test_losses
    )

    metrics = calculate_metrics(
        all_targets,
        all_preds
    )

    results = {

        "test_loss": test_loss,

        "acc": metrics["acc"],

        "f1_macro": metrics["f1_macro"],

        "f1_weighted": metrics["f1_weighted"],

        "confusion_matrix":
            metrics["confusion_matrix"],

        "y_true": all_targets,

        "y_pred": all_preds
    }

    return results