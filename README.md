# 🎵 RICONOSCIMENTO MALATTIE NEURODEGENERATIVE TRAMITE ANALISI DI SEGNALI AUDIO

Un framework completo per l'analisi di dati audio utilizzando tecniche di Deep Learning avanzate, con supporto per CNN, MLP classici, SVM e modelli Transformer pre-addestrati.

## 📋 Indice

- [Panoramica](#-panoramica)
- [Caratteristiche](#-caratteristiche)
- [Requisiti di Sistema](#-requisiti-di-sistema)
- [Installazione](#-installazione)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Configurazione](#-configurazione)
- [Utilizzo](#-utilizzo)
- [Dataset Supportati](#-dataset-supportati)
- [Modelli Disponibili](#-modelli-disponibili)
- [Risultati](#-risultati)
- [Risoluzione Problemi](#-risoluzione-problemi)

## 🎯 Panoramica

Il progetto implementa un sistema modulare per l'analisi e classificazione di dati audio, particolarmente orientato al riconoscimento di patologie vocali e stati emotivi. Supporta diversi approcci di machine learning:

- **CNN**: Reti neurali convoluzionali per analisi diretta del segnale
- **MLP Classici**: Multi-Layer Perceptron con feature tradizionali (MFCC, spettrali)
- **SVM**: Support Vector Machine con feature ingegnerizzate
- **Transformer+MLP**: Modelli pre-addestrati (Wav2Vec2, HuBERT, WavLM) + classificatore MLP

## ✨ Caratteristiche

- 🔧 **Configurazione YAML**: Setup completo tramite file di configurazione
- 📊 **Metriche Complete**: Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity
- 📈 **Visualizzazioni**: Grafici automatici delle metriche e curve di apprendimento
- 🎛️ **Preprocessing Avanzato**: Normalizzazione, resampling, padding automatico
- 🚀 **Ottimizzazioni**: DataLoader paralleli, GPU acceleration, mixed precision
- 💾 **Salvataggio Risultati**: Export automatico in JSON con metriche dettagliate

### Dipendenze Python
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- Librosa, scikit-learn, pandas

## 🚀 Installazione

### Setup Rapido
```bash
# Clona il repository
git clone <repository-url>
cd ProgettoTesi

# Installa dipendenze
bash prepare.sh

# Configura il dataset (vedi sezione Dataset)
mkdir -p dataset
# Posiziona i tuoi dataset nella cartella dataset/
```

### Setup Manuale
```bash
# Installa dipendenze Python
pip install -r requirements.txt

# Per sistemi Ubuntu (se problemi con tkinter)
sudo apt-get install python3-tk
```

## 📁 Struttura del Progetto

```
ProgettoTesi/
├── 📄 train.py                    # Script principale di training
├── 📄 metrics.py                  # Definizioni metriche
├── 📄 save_model_results.py       # Salvataggio risultati
├── 📄 prepare.sh                  # Script setup automatico
├── 📄 requirements.txt            # Dipendenze Python
├── 📁 config/
│   └── base_config.yaml          # Configurazione principale
├── 📁 data_classes/
│   └── dataset.py                # Gestione dataset audio
├── 📁 model_classes/
│   ├── cnn_model.py              # Modello CNN
│   └── mlp.py                    # Modello MLP
├── 📁 extract_representetion/
│   ├── classical_features.py     # Estrazione feature tradizionali
│   └── transformers_features.py  # Estrazione feature Transformer
├── 📁 visualization/
│   └── metrics_plots.py          # Visualizzazioni metriche
├── 📁 results/                   # Risultati salvati
│   ├── ita_pvs/                 # Risultati dataset Ita-PVS
│   ├── neurovoz/                # Risultati dataset Neurovoz
│   └── addresso/                # Risultati dataset Addresso
└── 📁 dataset/                   # Directory dataset (da creare)
    ├── Ita-PVS/
    ├── Neurovoz/
    └── Addresso/
```

## ⚙️ Configurazione

Il file `config/base_config.yaml` contiene tutti i parametri configurabili:

### Sezione Data
```yaml
data:
  train_ratio: 0.85              # Proporzione training set
  test_val_ratio: 0.35           # Proporzione test/validation
  data_dir: /path/to/dataset     # Percorso dataset
  dataset_name: "Addresso"       # Nome dataset
  target_sr: 16000               # Sample rate target
  fixed_duration_s: 5.0          # Durata fissa audio (secondi)
```

### Sezione Model
```yaml
model:
  branch: "transformers_mlp"     # Tipo modello
  mlp:
    hidden_layers: [256, 128, 64] # Architettura MLP
    dropout: 0.2                  # Dropout rate
    class_weight: "balanced"      # Bilanciamento classi
```

### Sezione Training
```yaml
training:
  epochs: 15                     # Numero epoche
  batch_size: 16                 # Dimensione batch
  lr: 0.0005                     # Learning rate
  device: "cuda"                 # Device (cuda/mps/cpu)
```

## 🎮 Utilizzo

### Training Base
```bash
# Training con configurazione default
python train.py

# Training con dataset specifico
python train.py --data.dataset_name "Ita-PVS"

# Training con modello specifico
python train.py --model.branch "cnn"
```

### Training Avanzato
```bash
# CNN con input spettrogramma
python train.py --model.branch "cnn" --model.cnn.in_type "spectrogram"

# Transformer specifico
python train.py --model.branch "transformers_mlp" \
  --features.transformers.model_name "facebook/hubert-base-ls960"

# SVM classico
python train.py --model.branch "classical_svm"
```

### Override Configurazione
```bash
# Modifica parametri via command line
python train.py --config config/custom_config.yaml \
  --data.dataset_name "Neurovoz" \
  --model.branch "transformers_mlp"
```

## 📊 Dataset Testati

### 1. Ita-PVS (Italian Parkinson Voice Samples)
- **Classi**: Healthy Control, Parkinson
- **Formato**: File .wav
- **Struttura**: Cartelle separate per classe

### 2. Neurovoz
- **Classi**: Healthy Control (HC_*), Parkinson (PD_*)
- **Formato**: File .wav
- **Identificazione**: Prefisso filename

### 3. Addresso (Alzheimer Detection)
- **Classi**: Healthy, Alzheimer
- **Formato**: File .wav
- **Caratteristica**: Split train/test predefiniti

### Aggiungere Nuovo Dataset
1. Posiziona i file nella cartella `dataset/NomeDataset/`
2. Aggiungi configurazione in `AudioDataset.DATASET_CONFIGS`
3. Definisci `label_extractor` per identificazione automatica classi

## 🤖 Modelli Disponibili

### 1. CNN (Convolutional Neural Network)
- **Input**: Waveform raw o spettrogramma
- **Architettura**: Conv1D + Pooling + FC layers
- **Vantaggi**: Veloce, interpretabile
- **Uso**: `--model.branch "cnn"`

### 2. MLP Classico
- **Features**: MFCC, spettrali, prosodiche
- **Architettura**: Fully connected layers
- **Vantaggi**: Leggero, feature interpretabili
- **Uso**: `--model.branch "classical_mlp"`

### 3. SVM (Support Vector Machine)
- **Features**: Stesse del MLP classico
- **Kernel**: RBF, Linear, Polynomial
- **Vantaggi**: Robusto, teoricamente fondato
- **Uso**: `--model.branch "classical_svm"`

### 4. Transformer + MLP
- **Backbone**: Wav2Vec2, HuBERT, WavLM
- **Head**: MLP classificatore
- **Vantaggi**: State-of-the-art performance
- **Modelli supportati**:
  - `facebook/wav2vec2-base-960h`
  - `facebook/hubert-base-ls960`
  - `microsoft/wavlm-base`
  - `ALM/wav2vec2-base-audioset`

## 📈 Risultati

I risultati vengono salvati automaticamente in `results/dataset_name/model_name/`:

### Metriche Salvate
- **Accuracy**: Precisione complessiva
- **Precision/Recall**: Per classe e macro/weighted average
- **F1-Score**: Armonica di precision/recall
- **Sensitivity**: True Positive Rate
- **Specificity**: True Negative Rate
- **Confusion Matrix**: Matrice di confusione
- **Classification Report**: Report dettagliato scikit-learn

### Visualizzazioni
- Curve di apprendimento (loss/accuracy)
- Matrice di confusione
- Metriche per classe
- Distribuzione predizioni

### Esempio Risultati
```json
{
  "model_name": "Transformers+MLP_facebook_hubert-base-ls960",
  "dataset_name": "Ita-PVS",
  "metrics": {
    "accuracy": 0.8228,
    "precision": 0.8584,
    "recall": 0.8228,
    "f1_score": 0.8144,
    "sensitivity": 0.9885,
    "specificity": 0.6197
  }
}
```

## 🔧 Risoluzione Problemi

### Errore tkinter (Ubuntu/WSL)
```bash
sudo apt-get install python3-tk
```

### Errore CUDA Out of Memory
- Riduci `batch_size` in configurazione
- Usa `device: "cpu"` per training CPU-only

### Dataset non trovato
- Verifica percorso in `data.data_dir`
- Controlla struttura cartelle dataset
- Assicurati che `dataset_name` corrisponda alla cartella

### Modello Transformer non scaricato
```bash
# Pre-download modelli
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/hubert-base-ls960')"
```

### Performance lente
- Abilita `num_workers` nei DataLoader
- Usa `pin_memory=True` per GPU
- Considera `mixed_precision` training

## 📝 Note Aggiuntive

- **Seed Riproducibilità**: Impostato automaticamente per risultati consistenti
- **Checkpoints**: Modelli salvati in `checkpoints/` dopo training
- **Logs**: Output dettagliato con progress bar e metriche real-time
- **Configurazione Flessibile**: Override parametri via command line o file YAML personalizzati

**Sviluppato per analisi audio avanzata con focus su applicazioni biomediche e riconoscimento malattie neurodegenerative.**