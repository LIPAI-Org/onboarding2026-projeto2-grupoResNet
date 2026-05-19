""" Onde o treino ocorre """

import os
import random
import numpy as np
import torch

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    seed=42,
    checkpoint_path=None,
    scheduler=None
):

    set_seed(seed)

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

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            train_losses.append(
                loss.item()
            )

            train_preds.extend(
                preds.cpu().numpy()
            )

            train_targets.extend(
                labels.cpu().numpy()
            )

        train_loss = np.mean(
            train_losses
        )

        train_acc = accuracy_score(
            train_targets,
            train_preds
        )

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

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                val_losses.append(
                    loss.item()
                )

                val_preds.extend(
                    preds.cpu().numpy()
                )

                val_targets.extend(
                    labels.cpu().numpy()
                )

        val_loss = np.mean(
            val_losses
        )

        val_acc = accuracy_score(
            val_targets,
            val_preds
        )

        if scheduler is not None:

            scheduler.step()

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_acc"].append(
            val_acc
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = epoch + 1

            if checkpoint_path is not None:

                checkpoint_dir = os.path.dirname(
                    checkpoint_path
                )

                if checkpoint_dir != "":

                    os.makedirs(
                        checkpoint_dir,
                        exist_ok=True
                    )

                torch.save(
                    {

                        "epoch": best_epoch,

                        "seed": seed,

                        "best_val_acc":
                            best_val_acc,

                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "history":
                            history
                    },
                    checkpoint_path
                )

    results = {

        "history": history,

        "best_val_acc": best_val_acc,

        "best_epoch": best_epoch
    }

    return results