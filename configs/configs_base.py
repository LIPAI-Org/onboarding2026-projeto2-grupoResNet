""" Define os valores padrões """
import torch

BATCH_SIZE = 32
NUM_EPOCAS = 50
TAXA_APRENDIZADO = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
