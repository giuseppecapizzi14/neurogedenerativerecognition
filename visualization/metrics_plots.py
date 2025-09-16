#!/usr/bin/env python3
"""
Modulo semplificato per la visualizzazione delle metriche di valutazione dei modelli.
Contiene solo confusion matrix e curva ROC.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class MetricsVisualizer:
    """
    Classe semplificata per la visualizzazione delle metriche di valutazione dei modelli.
    """
    
    def __init__(self, save_dir: str = "plots", figsize: tuple = (10, 8)):
        """
        Inizializza il visualizzatore delle metriche.
        
        Args:
            save_dir: Directory dove salvare i grafici
            figsize: Dimensioni della figura (larghezza, altezza)
        """
        self.save_dir = save_dir
        self.figsize = figsize
        
        # Crea la directory se non esiste
        os.makedirs(save_dir, exist_ok=True)
        
        # Imposta lo stile dei grafici
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, 
                            y_true: List[int], 
                            y_pred: List[int],
                            class_names: Optional[List[str]] = None,
                            title: str = "Matrice di Confusione",
                            save_name: Optional[str] = None,
                            model_dir: Optional[str] = None) -> None:
        """
        Crea una matrice di confusione.
        
        Args:
            y_true: Etichette vere
            y_pred: Predizioni del modello
            class_names: Nomi delle classi
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
            model_dir: Directory specifica del modello (se None usa save_dir)
        """
        # Calcola la matrice di confusione
        cm = confusion_matrix(y_true, y_pred)
        
        # Crea il grafico
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Numero di Campioni'})
        
        # Personalizzazione
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predizione', fontsize=12, fontweight='bold')
        ax.set_ylabel('Etichetta Vera', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Salva il grafico
        if save_name:
            save_dir = model_dir if model_dir else self.save_dir
            save_path = os.path.join(save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Matrice di confusione salvata: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, 
                      y_true: List[int], 
                      y_scores: List[float],
                      title: str = "Curva ROC",
                      save_name: Optional[str] = None,
                      model_dir: Optional[str] = None) -> None:
        """
        Crea una curva ROC.
        
        Args:
            y_true: Etichette vere (0 o 1)
            y_scores: Punteggi di probabilità per la classe positiva
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
            model_dir: Directory specifica del modello (se None usa save_dir)
        """
        # Calcola la curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Crea il grafico
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Curva ROC
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'Curva ROC (AUC = {roc_auc:.3f})')
        
        # Linea diagonale (classificatore casuale)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Classificatore Casuale (AUC = 0.500)')
        
        # Personalizzazione
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasso di Falsi Positivi', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tasso di Veri Positivi', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva il grafico
        if save_name:
            save_dir = model_dir if model_dir else self.save_dir
            save_path = os.path.join(save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Curva ROC salvata: {save_path}")
        
        plt.close()