""" Algumas funções reutilizáveis """
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    
    f1_macro = f1_score(
        y_true, 
        y_pred, 
        average="macro"
    )
    
    f1_weighted = f1_score(
        y_true, 
        y_pred, 
        average="weighted"
    )
    
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm
    }

    return results