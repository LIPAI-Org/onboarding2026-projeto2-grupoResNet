""" Especificações da ResNet18 """

import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


def create_resnet18(num_classes, training_mode):

    training_mode = training_mode.lower()

    if training_mode == "fs":
        model = resnet18(weights=None)

    elif training_mode in ["pt_fc", "pt_all"]:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

    else:
        raise ValueError(f"modo de treino invalido: {training_mode}")

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    if training_mode == "pt_fc":

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif training_mode == "pt_all":

        for param in model.parameters():
            param.requires_grad = True

    return model