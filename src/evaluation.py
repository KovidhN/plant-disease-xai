import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def generate_confusion_matrix(y_true, y_pred, labels=None):
    """
    Returns the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
