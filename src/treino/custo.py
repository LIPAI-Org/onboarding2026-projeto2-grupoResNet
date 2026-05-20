""" Responsável pelos calculos de custo durante o treinamento """

def count_parameters(model):
    return sum(
        p.numel() 
        for p in model.parameters()
    )


def count_trainable_parameters(model):
    return sum(
        p.numel() 
        for p in model.parameters() 
        if p.requires_grad
    )


def estimate_gflops(model):
    model_name = model.__class__.__name__.lower()

    gflops_table = {
        "resnet": 1.8
    }

    for key in gflops_table:
        if key in model_name:
            return gflops_table[key]

    return None