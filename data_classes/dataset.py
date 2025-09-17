import os
import numpy as np
import pandas as pd
from typing import TypedDict, Dict, Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Resample, MelSpectrogram


class Sample(TypedDict):
    waveform: Tensor
    label: int

def apply_preprocessing(waveform: Tensor, sr: int, cfg: Dict[str, Any]) -> Tensor:
    """Preprocessing unificato: mono -> resample -> normalize -> crop/pad"""
    # Mono conversion
    if cfg['data']['mono'] and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample
    if cfg['data']['resample'] and sr != cfg['data']['target_sr']:
        resampler = Resample(orig_freq=sr, new_freq=cfg['data']['target_sr'])
        waveform = resampler(waveform)
    
    # Normalize
    if cfg['data']['normalize'] == 'peak':
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    elif cfg['data']['normalize'] == 'rms':
        rms = torch.sqrt(torch.mean(waveform**2))
        waveform = waveform / (rms + 1e-8)
    
    # Fixed duration (crop/pad)
    target_length = int(cfg['data']['fixed_duration_s'] * cfg['data']['target_sr'])
    current_length = waveform.shape[1]
    
    if current_length > target_length:
        # Crop
        waveform = waveform[:, :target_length]
    elif current_length < target_length:
        # Pad with silence
        padding = target_length - current_length
        waveform = F.pad(waveform, (0, padding), mode='constant', value=0)
    
    return waveform


class AudioDataset(Dataset[Sample]):
    # Dataset specifici
    DATASET_CONFIGS = {
        "Ita-PVS": {
            "label_dict": {"healthy": 0, "parkinson": 1},
            "file_pattern": ".wav",
            "label_extractor": lambda filepath: "healthy" if "Healthy Control" in filepath else "parkinson"
        },
        "Neurovoz": {
            "label_dict": {"healthy": 0, "parkinson": 1},
            "file_pattern": ".wav", 
            "label_extractor": lambda filepath: "healthy" if os.path.basename(filepath).startswith("HC_") else "parkinson"
        },
        "Addresso": {
            "label_dict": {"healthy": 0, "parkinson": 1},
            "file_pattern": ".wav",
            "label_extractor": lambda filename: "healthy" if "healthy" in filename else "parkinson"
        }
    }

    def __init__(self, cfg: Dict[str, Any], split: str = 'train'):
        self.cfg = cfg
        self.split = split
        self.dataset_name = cfg['data']['dataset_name']
        self.data_dir = os.path.join(cfg['data']['data_dir'], self.dataset_name)
        
        # Setup dataset specifico
        if self.dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Dataset {self.dataset_name} non supportato")
        
        self.dataset_config = self.DATASET_CONFIGS[self.dataset_name]
        
        # MelSpectrogram transform (se necessario per CNN)
        self.mel_transform = None
        if cfg['model']['branch'] == 'cnn' and cfg['model']['cnn']['in_type'] == 'spectrogram':
            self.mel_transform = MelSpectrogram(
                sample_rate=cfg['data']['target_sr'],
                n_mels=cfg['features']['spectrogram']['n_mels'],
                n_fft=cfg['features']['spectrogram']['n_fft'],
                hop_length=cfg['features']['spectrogram']['hop_length']
            )
        
        # Carica o genera labels.csv
        self._load_or_generate_labels()

    def _load_or_generate_labels(self):
        """Carica o genera il file labels.csv"""
        labels_path = os.path.join(self.data_dir, 'labels.csv')
        print(f"Looking for labels file at: {labels_path}")
        print(f"Labels file exists: {os.path.exists(labels_path)}")
        
        if os.path.exists(labels_path):
            # Carica labels esistenti
            df = pd.read_csv(labels_path)
            # Normalizza i path separators per compatibilità cross-platform
            self.audio_files = [os.path.normpath(filepath.replace('\\', '/')) for filepath in df['filepath'].tolist()]
            self.labels = df['label_id'].tolist()
            print(f"✅ Preprocessing completato con successo: caricati {len(self.audio_files)} file audio dal file labels esistente.")
        else:
            # Genera labels.csv
            self.audio_files = []
            self.labels = []
            label_texts = []
            
            print(f"Scanning dataset directory: {self.data_dir}")
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(self.dataset_config['file_pattern']):
                        filepath = os.path.join(root, file)
                        # Normalizza i path separators per compatibilità cross-platform
                        filepath = os.path.normpath(filepath.replace('\\', '/'))
                        
                        # Estrai label dal filepath completo
                        label_text = self.dataset_config['label_extractor'](filepath)
                        if label_text in self.dataset_config['label_dict']:
                            label_id = self.dataset_config['label_dict'][label_text]
                            
                            self.audio_files.append(filepath)
                            self.labels.append(label_id)
                            label_texts.append(label_text)
            
            print(f"Found {len(self.audio_files)} files")
            
            if len(self.audio_files) > 0:
                # Salva labels.csv
                df = pd.DataFrame({
                    'filepath': self.audio_files,
                    'label_text': label_texts,
                    'label_id': self.labels
                })
                df.to_csv(labels_path, index=False)
            else:
                print("❌ Preprocessing fallito: nessun file audio trovato nel dataset.")
                raise ValueError(f"Nessun file audio trovato nel dataset {self.dataset_name}")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Sample:
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Normalizza il path per compatibilità cross-platform (sicurezza aggiuntiva)
        audio_path = os.path.normpath(audio_path.replace('\\', '/'))
        
        # Debug: stampa il path che sta cercando di caricare
        print(f"Tentativo di caricare: {audio_path}")
        print(f"File esiste: {os.path.exists(audio_path)}")
        
        # Carica audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Preprocessing unificato per tutti i modelli
        waveform = apply_preprocessing(waveform, sample_rate, self.cfg)
        
        # Prepara output base
        sample = {
            "waveform": waveform,
            "label": label
        }
        
        # Calcola spettrogramma se necessario per CNN
        if self.mel_transform is not None:
            mel_spec = self.mel_transform(waveform)
            if self.cfg['features']['spectrogram']['log']:
                mel_spec = torch.log(mel_spec + 1e-8)
            sample["waveform"] = mel_spec
        
        return sample
