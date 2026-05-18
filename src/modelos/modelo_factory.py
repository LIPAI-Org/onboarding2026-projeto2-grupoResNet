""" Factory de modelos, facilita criação :) """

from .resnet18 import create_resnet18
from .resnet34 import create_resnet34


def get_model(
    model_name,
    num_classes,
    training_mode="fs"
):

    model_name = model_name.lower()

    if model_name == "resnet18":

        return create_resnet18(
            num_classes=num_classes,
            training_mode=training_mode
        )

    elif model_name == "resnet34":

        return create_resnet34(
            num_classes=num_classes,
            training_mode=training_mode
        )

    else:
        raise ValueError(f"modelo invalido: {model_name}")