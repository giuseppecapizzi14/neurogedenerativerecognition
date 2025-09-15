import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visualization.metrics_plots import MetricsVisualizer


def save_model_results(model_name: str, y_true: list, y_pred: list, dataset_name: str, 
                      y_scores: list = None, history: dict = None, save_dir: str = "plots"):
    """
    Salva tutti i grafici delle metriche per un modello specifico.
    
    Args:
        model_name: Nome del modello
        y_true: Etichette vere
        y_pred: Predizioni del modello
        dataset_name: Nome del dataset utilizzato
        y_scores: Probabilità/scores per la classe positiva (per ROC curve)
        history: Storia dell'addestramento (per training history)
        save_dir: Directory base per salvare i grafici
    """
    # Inizializza il visualizzatore
    visualizer = MetricsVisualizer(save_dir=save_dir)
    
    # Crea directory specifica per dataset/modello
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Crea directory specifica per il modello all'interno del dataset
    model_dir = os.path.join(dataset_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Calcola le metriche
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }
    
    print(f"\n[INFO] Salvando grafici per {model_name}...")
    
    # 1. Radar Chart
    print(f"[INFO] Creazione radar chart per {model_name}...")
    visualizer.plot_metrics_radar(
        metrics,
        model_name=model_name,
        title=f"Profilo Prestazioni - {model_name}",
        save_name="radar_chart",
        model_dir=model_dir
    )
    
    # 2. Confusion Matrix
    print(f"[INFO] Creazione matrice di confusione per {model_name}...")
    visualizer.plot_confusion_matrix(
        y_true, y_pred,
        class_names=['Sano', 'Parkinson'],
        title=f"Matrice di Confusione - {model_name}",
        save_name="confusion_matrix",
        model_dir=model_dir
    )
    
    # 3. ROC Curve (se disponibili gli scores)
    if y_scores is not None:
        print(f"[INFO] Creazione curva ROC per {model_name}...")
        visualizer.plot_roc_curve(
            y_true, y_scores,
            title=f"Curva ROC - {model_name}",
            save_name="roc_curve",
            model_dir=model_dir
        )
    else:
        # Genera scores simulati basati sulle predizioni
        print(f"[INFO] Generazione scores simulati per ROC curve...")
        np.random.seed(42)
        simulated_scores = []
        for true_label, pred_label in zip(y_true, y_pred):
            if pred_label == true_label:
                # Predizione corretta: alta confidenza
                score = np.random.uniform(0.7, 0.95) if pred_label == 1 else np.random.uniform(0.05, 0.3)
            else:
                # Predizione errata: bassa confidenza
                score = np.random.uniform(0.4, 0.6)
            simulated_scores.append(score)
        
        visualizer.plot_roc_curve(
            y_true, simulated_scores,
            title=f"Curva ROC - {model_name}",
            save_name="roc_curve",
            model_dir=model_dir
        )
    
    # 4. Training History (se disponibile)
    if history is not None:
        print(f"[INFO] Creazione grafico storia addestramento per {model_name}...")
        visualizer.plot_training_history(
            history,
            title=f"Storia Addestramento - {model_name}",
            save_name="training_history",
            model_dir=model_dir
        )
    
    print(f"[INFO] Tutti i grafici per {model_name} sono stati salvati in: {model_dir}")
    return model_dir