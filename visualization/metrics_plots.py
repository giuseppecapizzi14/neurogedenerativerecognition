#!/usr/bin/env python3
"""
Modulo per la visualizzazione delle metriche di valutazione dei modelli.
Contiene la classe MetricsVisualizer per creare grafici di confronto, radar chart,
matrici di confusione e curve ROC.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class MetricsVisualizer:
    """
    Classe per la visualizzazione delle metriche di valutazione dei modelli.
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
    
    def plot_metrics_comparison(self, 
                              metrics_dict: Dict[str, Dict[str, float]], 
                              title: str = "Confronto Metriche Modelli",
                              save_name: Optional[str] = None) -> None:
        """
        Crea un grafico a barre per confrontare le metriche di più modelli.
        
        Args:
            metrics_dict: Dizionario {nome_modello: {metrica: valore}}
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
        """
        # Converti in DataFrame per facilità di plotting
        df = pd.DataFrame(metrics_dict).T
        
        # Crea il grafico
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Grafico a barre raggruppate
        df.plot(kind='bar', ax=ax, width=0.8)
        
        # Personalizzazione
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Modelli', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valore Metrica', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(title='Metriche', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Ruota le etichette dell'asse x
        plt.xticks(rotation=45, ha='right')
        
        # Aggiungi valori sopra le barre
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9)
        
        plt.tight_layout()
        
        # Salva il grafico
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Grafico salvato: {save_path}")
        
        plt.close()
    
    def plot_metrics_radar(self, 
                          metrics: Dict[str, float],
                          model_name: str = "Modello",
                          title: Optional[str] = None,
                          save_name: Optional[str] = None) -> None:
        """
        Crea un radar chart per visualizzare le metriche di un singolo modello.
        
        Args:
            metrics: Dizionario delle metriche {nome_metrica: valore}
            model_name: Nome del modello
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
        """
        # Prepara i dati
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Numero di variabili
        N = len(categories)
        
        # Calcola gli angoli per ogni asse
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Completa il cerchio
        
        # Aggiungi il primo valore alla fine per chiudere il poligono
        values += values[:1]
        
        # Crea il grafico
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection='polar'))
        
        # Disegna il poligono
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
        
        # Aggiungi le etichette
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Imposta i limiti dell'asse radiale
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Titolo
        if title is None:
            title = f"Radar Chart - {model_name}"
        ax.set_title(title, size=16, fontweight='bold', pad=20)
        
        # Legenda
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Salva il grafico
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Radar chart salvato: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, 
                            y_true: List[int], 
                            y_pred: List[int],
                            class_names: Optional[List[str]] = None,
                            title: str = "Matrice di Confusione",
                            save_name: Optional[str] = None) -> None:
        """
        Crea una matrice di confusione.
        
        Args:
            y_true: Etichette vere
            y_pred: Predizioni del modello
            class_names: Nomi delle classi
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
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
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎯 Matrice di confusione salvata: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, 
                      y_true: List[int], 
                      y_scores: List[float],
                      title: str = "Curva ROC",
                      save_name: Optional[str] = None) -> None:
        """
        Crea una curva ROC.
        
        Args:
            y_true: Etichette vere (0 o 1)
            y_scores: Punteggi di probabilità per la classe positiva
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
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
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Curva ROC salvata: {save_path}")
        
        plt.close()
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            title: str = "Storia dell'Addestramento",
                            save_name: Optional[str] = None) -> None:
        """
        Crea grafici per visualizzare la storia dell'addestramento.
        
        Args:
            history: Dizionario con le metriche per epoca {metrica: [valori]}
            title: Titolo del grafico
            save_name: Nome del file da salvare (senza estensione)
        """
        # Numero di metriche
        n_metrics = len(history)
        
        # Crea subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot per ogni metrica
        for i, (metric, values) in enumerate(history.items()):
            epochs = range(1, len(values) + 1)
            axes[i].plot(epochs, values, 'b-', linewidth=2, marker='o')
            axes[i].set_title(f'{metric.capitalize()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Epoca', fontsize=12)
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        # Titolo generale
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Salva il grafico
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Storia addestramento salvata: {save_path}")
        
        plt.close()