#!/usr/bin/env python3
"""
Modulo semplificato per salvare i risultati dei modelli con metriche essenziali.
Include accuracy, precision, recall, F1 Score, sensitivity, specificity, confusion matrix e curva ROC.
"""

import os
import json
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from visualization.metrics_plots import MetricsVisualizer


def calculate_sensitivity_specificity(y_true: List[int], y_pred: List[int]) -> tuple:
    """
    Calcola sensitivity (sensibilità) e specificity (specificità).
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni del modello
        
    Returns:
        tuple: (sensitivity, specificity)
    """
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)
    
    # Per classificazione binaria
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (True Positive Rate) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (True Negative Rate) = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
    else:
        # Per classificazione multi-classe, calcola la media
        sensitivity_per_class = []
        specificity_per_class = []
        
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivity_per_class.append(sens)
            specificity_per_class.append(spec)
        
        sensitivity = np.mean(sensitivity_per_class)
        specificity = np.mean(specificity_per_class)
    
    return sensitivity, specificity


def save_model_results(model_name: str,
                      y_true: List[int],
                      y_pred: List[int],
                      dataset_name: str,
                      y_scores: Optional[List[float]] = None,
                      class_names: Optional[List[str]] = None,
                      save_dir: str = "results") -> Dict[str, Any]:
    """
    Salva i risultati del modello con metriche essenziali e grafici.
    Organizza i risultati per dataset e poi per modello: results/dataset_name/model_name/
    
    Args:
        model_name: Nome del modello (es. "SVM", "MLP", "CNN")
        y_true: Etichette vere
        y_pred: Predizioni del modello
        dataset_name: Nome del dataset (es. "Ita-PVS", "MDVR-KCL", etc.)
        y_scores: Punteggi di probabilità (per curva ROC)
        class_names: Nomi delle classi
        save_dir: Directory base dove salvare i risultati
        
    Returns:
        Dict: Dizionario con tutte le metriche calcolate
    """
    print(f"\n[INFO] Salvando risultati per {model_name} su dataset {dataset_name}")
    
    # Crea struttura directory: results/dataset_name/model_name/
    dataset_dir = os.path.join(save_dir, dataset_name.lower().replace(' ', '_').replace('-', '_'))
    model_dir = os.path.join(dataset_dir, model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    # Calcola le metriche principali
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calcola sensitivity e specificity
    sensitivity, specificity = calculate_sensitivity_specificity(y_true, y_pred)
    
    # Crea il dizionario delle metriche
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    print(f"[INFO] Metriche calcolate:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Sensitivity: {sensitivity:.4f}")
    print(f"  - Specificity: {specificity:.4f}")
    
    # Salva il classification report completo
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Salva le metriche in formato JSON
    results_data = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'metrics': metrics,
        'classification_report': report
    }
    
    metrics_file = os.path.join(model_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metriche salvate in: {metrics_file}")
    
    # Crea i grafici
    visualizer = MetricsVisualizer(save_dir=model_dir)
    
    # 1. Confusion Matrix
    visualizer.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title=f"Matrice di Confusione - {model_name} ({dataset_name})",
        save_name="confusion_matrix",
        model_dir=model_dir
    )
    
    # 2. Curva ROC (solo se abbiamo i punteggi)
    if y_scores is not None:
        # Per classificazione binaria
        if len(set(y_true)) == 2:
            visualizer.plot_roc_curve(
                y_true=y_true,
                y_scores=y_scores,
                title=f"Curva ROC - {model_name} ({dataset_name})",
                save_name="roc_curve",
                model_dir=model_dir
            )
        else:
            print("[WARNING] Curva ROC disponibile solo per classificazione binaria")
    else:
        print("[WARNING] Punteggi non forniti, curva ROC non generata")
    
    print(f"[INFO] Risultati salvati in: {model_dir}")
    print(f"[INFO] Struttura: {save_dir}/{dataset_name}/{model_name}/")
    return results_data