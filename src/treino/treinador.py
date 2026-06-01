""" Onde o treino ocorre """

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import configs.configs_base as cb
from src.utils.seed import definir_seed
from src.utils.checkpoints import save_checkpoint
from configs.datasets.base import DatasetConfig


def train_model(
    model,
    train_loader,
    val_loader,
    config_dataset: DatasetConfig,
    seed=42,
    scheduler=None,
    checkpoint_path=None
):
    definir_seed(seed)
    device = cb.DEVICE
    num_epochs = cb.NUM_EPOCAS
    
    optimizer = cb.OTIMIZADOR(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cb.TAXA_APRENDIZADO
    )

    if config_dataset.is_multi:
        criterion = cb.CRITERION_MULTI
    else:
        criterion = cb.CRITERION_BIN

    model = model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # FASE DE TREINO
       
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch+1}/{num_epochs} train"
        )

        for images, labels in train_bar:
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

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if config_dataset.is_multi:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = (torch.sigmoid(outputs) >= 0.5).long()

            train_losses.append(loss.item())
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_targets, train_preds)

        # FASE DE VALIDAÇÃO
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"epoch {epoch+1}/{num_epochs} val"
            )

            for images, labels in val_bar:
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

                val_losses.append(loss.item())
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            if checkpoint_path is not None:
                save_checkpoint(
                    path=checkpoint_path,
                    modelo=model,
                    otim=optimizer,
                    epoca=best_epoch,
                    best_val_acc=best_val_acc,
                    extra={
                        "seed": seed,
                        "dataset": getattr(config_dataset, "nome", None)
                    }
                )

    results = {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "otimizador": optimizer
    }

    return results