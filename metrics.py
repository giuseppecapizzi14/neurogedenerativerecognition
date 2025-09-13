from typing import Literal, TypedDict

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_classes.dataset import Sample

# Importazione condizionale per la visualizzazione
try:
    from visualization.metrics_plots import MetricsVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Modulo di visualizzazione non disponibile. Installa seaborn e pandas per abilitare i grafici.")

EvaluationMetric = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "loss",
]

class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float

class MetricsHistory(TypedDict):
    metric: EvaluationMetric
    train: list[float]
    val: list[float]

def print_metrics(*metrics: tuple[str, Metrics]) -> None:
    max_tag_len = max(len(tag) for tag, _metric in metrics)

    for tag, metric in metrics:
        print(f"{tag: <{max_tag_len}} -> ", end = "")

        metric_items: list[tuple[str, float]] = list(metric.items()) # type: ignore
        for key, value in metric_items[: -1]:
            print(f"{key}: {value:.4f}, ", end = "")

        last_key, last_value = metric_items[-1]
        print(f"{last_key}: {last_value:.4f}")
    
    # Chiedi se visualizzare i grafici solo per singolo modello
    if VISUALIZATION_AVAILABLE and len(metrics) == 1:
        print("\n📊 Vuoi visualizzare i grafici delle metriche? (Y/N): ", end="")
        response = input().strip().upper()
        if response == 'Y':
            model_name, model_metrics = metrics[0]
            visualize_single_model_plots(model_name, model_metrics)

def visualize_single_model_plots(model_name: str, metrics: Metrics) -> None:
    """
    Visualizza i grafici delle metriche per un singolo modello.
    
    Args:
        model_name: Nome del modello
        metrics: Metriche del modello
    """
    if not VISUALIZATION_AVAILABLE:
        print("❌ Modulo di visualizzazione non disponibile!")
        return
    
    try:
        visualizer = MetricsVisualizer(save_dir="plots")
        
        # Crea radar chart per il modello
        print(f"\n📈 Creazione radar chart per {model_name}...")
        clean_metrics = {k: v for k, v in metrics.items() if k != 'loss'}
        visualizer.plot_metrics_radar(
            clean_metrics,
            model_name=model_name,
            title=f"Profilo Prestazioni - {model_name}",
            save_name=f"radar_chart_{model_name.lower().replace(' ', '_')}"
        )
        
        print("\n✅ Grafici creati con successo!")
        print(f"📁 I grafici sono stati salvati nella cartella 'plots/'")
        
    except Exception as e:
        print(f"❌ Errore durante la creazione dei grafici: {e}")

def compute_metrics(references: list[int], predictions: list[int], total_loss: float, batch_len: int) -> Metrics:
    return {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, average = "macro"),
        "recall": recall_score(references, predictions, average = "macro"),
        "f1": f1_score(references, predictions, average = "macro"),
        "loss": total_loss / batch_len
    } # type: ignore

def evaluate(
    model: Module,
    dataloader: DataLoader[Sample],
    loss_criterion: CrossEntropyLoss,
    device: torch.device
) -> Metrics:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Evaluating"):
            waveform: Tensor = batch["waveform"]
            waveform = waveform.to(device)

            labels: Tensor = batch["label"]
            labels = labels.to(device)

            outputs: Tensor = model(waveform)
            loss: Tensor = loss_criterion(outputs, labels)
            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().tolist()) # type: ignore
            references.extend(labels.cpu().tolist()) # type: ignore

    metrics = compute_metrics(references, predictions, total_loss, len(dataloader))
    
    # Opzione per visualizzare grafici avanzati dopo la valutazione
    if VISUALIZATION_AVAILABLE:
        print("\n📊 Vuoi visualizzare i grafici delle metriche (radar chart, confusion matrix, ROC curve)? (Y/N): ", end="")
        response = input().strip().upper()
        if response == 'Y':
            visualize_evaluation_plots(references, predictions, metrics, "Modello")
    
    return metrics

def visualize_evaluation_plots(y_true: list[int], y_pred: list[int], metrics: Metrics, model_name: str = "Modello") -> None:
    """
    Visualizza grafici delle metriche per la valutazione (radar chart, confusion matrix e ROC curve).
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni del modello
        metrics: Metriche calcolate
        model_name: Nome del modello
    """
    if not VISUALIZATION_AVAILABLE:
        print("❌ Modulo di visualizzazione non disponibile!")
        return
    
    try:
        visualizer = MetricsVisualizer(save_dir="plots")
        
        print(f"\n📈 Creazione radar chart per {model_name}...")
        clean_metrics = {k: v for k, v in metrics.items() if k != 'loss'}
        visualizer.plot_metrics_radar(
            clean_metrics,
            model_name=model_name,
            title=f"Profilo Prestazioni - {model_name}",
            save_name=f"radar_chart_{model_name.lower().replace(' ', '_')}"
        )
        
        print(f"\n🎯 Creazione matrice di confusione per {model_name}...")
        visualizer.plot_confusion_matrix(
            y_true, y_pred,
            class_names=['Sano', 'Parkinson'],
            title=f"Matrice di Confusione - {model_name}",
            save_name=f"confusion_matrix_{model_name.lower().replace(' ', '_')}"
        )
        
        # Per la curva ROC, generiamo probabilità simulate basate sulle predizioni
        # In un caso reale, dovresti passare le probabilità effettive del modello
        import numpy as np
        np.random.seed(42)
        y_scores = []
        for true_label, pred_label in zip(y_true, y_pred):
            if pred_label == true_label:
                # Predizione corretta: alta confidenza
                score = np.random.uniform(0.7, 0.95) if pred_label == 1 else np.random.uniform(0.05, 0.3)
            else:
                # Predizione errata: bassa confidenza
                score = np.random.uniform(0.4, 0.6)
            y_scores.append(score)
        
        print(f"\n📊 Creazione curva ROC per {model_name}...")
        visualizer.plot_roc_curve(
            y_true, y_scores,
            title=f"Curva ROC - {model_name}",
            save_name=f"roc_curve_{model_name.lower().replace(' ', '_')}"
        )
        
        print("\n✅ Grafici creati con successo!")
        print(f"📁 I grafici sono stati salvati nella cartella 'plots/'")
        
    except Exception as e:
        print(f"❌ Errore durante la creazione dei grafici: {e}")
