""" Responsável pelo calculo das métricas de treinamento (acurácia, etc)"""

import numpy as np
import torch
from tqdm import tqdm

from src.treino.metricas import calculate_metrics
from src.utils.seed import definir_seed
from configs.datasets.base import DatasetConfig
import configs.configs_base as cb


def evaluate_model(
    model,
    test_loader,
    config_dataset: DatasetConfig,
    seed=42,
    checkpoint_path=None
):
    definir_seed(seed)
    device = cb.DEVICE
    
    if config_dataset.is_multi:
        criterion = cb.CRITERION_MULTI
    else:
        criterion = cb.CRITERION_BIN

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
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

            if config_dataset.is_binario:
                labels = labels.float()
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
            else:
                labels = labels.long()
                if labels.ndim > 1:
                    labels = labels.squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if config_dataset.is_multi:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = (torch.sigmoid(outputs) >= 0.5).long()

            test_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_loss = np.mean(test_losses)
    metrics = calculate_metrics(all_targets, all_preds)

    results = {
        "test_loss": test_loss,
        "acc": metrics["acc"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "confusion_matrix": metrics["confusion_matrix"],
        "y_true": all_targets,
        "y_pred": all_preds
    }

    return results