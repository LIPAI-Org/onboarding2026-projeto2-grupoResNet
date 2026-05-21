from src.data.datasets import datasets
import configs.datasets.base as base
from src.data.dataloaders import DataLoaders
from src.modelos.modelo_factory import get_model
from src.treino.treinador import train_model
from src.treino.avaliador import evaluate_model

data = base.DatasetConfig(
    tipo_tarefa="binario",
    nro_classes=1,
    nome="ROI",
    tam_input=(224, 224),
    canais_input=3,
    normalizacao_mean=(0.485, 0.456, 0.406),
    normalizacao_std=(0.229, 0.224, 0.225),
)

dataset = datasets(config=data, escolha_transformada="base")
dataloaders = DataLoaders(config=data, datasets=dataset)

train_loader, val_loader, test_loader = dataloaders.criar_dataloaders_base()

model = get_model(model_name="resnet18", num_classes=data.nro_classes, training_mode="fs")

results = train_model(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    config_dataset=data,  
    scheduler=None
)

teste_results = evaluate_model(
    model=model,
    test_loader=test_loader,
    config_dataset=data,
    seed=42,
)

print("\n================ RESULTADOS FINAIS DO TESTE ================")
print(f"Loss no Teste: {teste_results['test_loss']:.4f}")
print(f"Acurácia no Teste: {teste_results['acc']:.4f}")
print(f"F1-Score Macro: {teste_results['f1_macro']:.4f}")
print(f"F1-Score Weighted: {teste_results['f1_weighted']:.4f}")
print("\nMatriz de Confusão:")
print(teste_results['confusion_matrix'])
print("============================================================")
