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
    """
    Applica preprocessing unificato per tutti i modelli.
    Gestisce anche waveform vuoti o corrotti.
    """
    # Validazione input: controlla se il waveform è vuoto
    if waveform.numel() == 0:
        print("Warning: Empty waveform passed to preprocessing. Creating fallback.")
        target_length = int(cfg['data']['fixed_duration_s'] * cfg['data']['target_sr'])
        waveform = torch.zeros(1, target_length)
        return waveform
    
    # Mono conversion
    if cfg['data']['mono'] and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample - con validazione aggiuntiva
    if cfg['data']['resample'] and sr != cfg['data']['target_sr']:
        try:
            resampler = Resample(orig_freq=sr, new_freq=cfg['data']['target_sr'])
            waveform = resampler(waveform)
        except Exception as e:
            print(f"Warning: Resampling failed: {e}. Using original waveform.")
    
    # Normalize
    if cfg['data']['normalize'] == 'peak':
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-8:  # Evita divisione per zero
            waveform = waveform / max_val
    elif cfg['data']['normalize'] == 'rms':
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 1e-8:  # Evita divisione per zero
            waveform = waveform / rms
    
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
            "label_dict": {"cn": 0, "ad": 1},  # cn=controlli, ad=alzheimer
            "file_pattern": ".wav",
            "label_extractor": lambda filepath: "cn" if "/cn/" in filepath or "\\cn\\" in filepath else "ad",
            "has_predefined_splits": True,  # Indica che train/test sono già separati
            "has_segmentation": True  # Indica che usa file CSV per segmentazione
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
        
        # Inizializza liste per dati
        self.audio_files = []
        self.labels = []
        
        # Per Addresso, aggiungi supporto per segmentazione
        if self.dataset_config.get('has_segmentation', False):
            self.segments = []  # Lista di (start_ms, end_ms) per ogni campione
            self.speakers = []  # Lista di speaker ID per ogni campione
        
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
            
            # Controlla se il dataset ha split predefiniti ma il file non ha la colonna split
            if self.dataset_config.get('has_predefined_splits', False):
                if 'split' not in df.columns:
                    print(f"⚠️  File labels.csv esistente non ha colonna 'split'. Rigenerazione necessaria...")
                    # Elimina il file esistente per forzare la rigenerazione
                    os.remove(labels_path)
                    # Ricorsione per rigenerare
                    self._load_or_generate_labels()
                    return
                else:
                    # Filtra per split se la colonna esiste
                    df = df[df['split'] == self.split]
            
            # Normalizza i path separators per compatibilità cross-platform
            self.audio_files = [os.path.normpath(filepath.replace('\\', '/')) for filepath in df['filepath'].tolist()]
            self.labels = df['label_id'].tolist()
        else:
            # Genera labels.csv
            self.audio_files = []
            self.labels = []
            label_texts = []
            splits = []
            
            print(f"Scanning dataset directory: {self.data_dir}")
            
            # Gestione speciale per dataset con split predefiniti (come Addresso)
            if self.dataset_config.get('has_predefined_splits', False):
                # Scansiona train e test separatamente
                split_dirs = {'train': 'train', 'test': 'test'}
                split_dir = os.path.join(self.data_dir, split_dirs[self.split])
                
                if os.path.exists(split_dir):
                    # Per Addresso con segmentazione
                    if self.dataset_config.get('has_segmentation', False):
                        self._process_addresso_segmentation(split_dir, splits)
                    else:
                        # Gestione normale per altri dataset con split predefiniti
                        for root, _, files in os.walk(split_dir):
                            for file in files:
                                if file.endswith(self.dataset_config['file_pattern']):
                                    filepath = os.path.join(root, file)
                                    filepath = os.path.normpath(filepath.replace('\\', '/'))
                                    
                                    label_text = self.dataset_config['label_extractor'](filepath)
                                    if label_text in self.dataset_config['label_dict']:
                                        label_id = self.dataset_config['label_dict'][label_text]
                                        
                                        self.audio_files.append(filepath)
                                        self.labels.append(label_id)
                                        label_texts.append(label_text)
                                        splits.append(self.split)
            else:
                # Gestione normale per dataset senza split predefiniti
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
                                splits.append('all')  # Sarà gestito dal train_test_split
            
            print(f"Found {len(self.audio_files)} files")
            
            if len(self.audio_files) > 0:
                # Salva labels.csv
                df_data = {
                    'filepath': self.audio_files,
                    'label_text': [list(self.dataset_config['label_dict'].keys())[list(self.dataset_config['label_dict'].values()).index(label)] for label in self.labels],
                    'label_id': self.labels
                }
                
                # Aggiungi colonne aggiuntive se necessario
                if self.dataset_config.get('has_predefined_splits', False):
                    df_data['split'] = splits
                
                if self.dataset_config.get('has_segmentation', False):
                    df_data['start_ms'] = [seg[0] for seg in self.segments]
                    df_data['end_ms'] = [seg[1] for seg in self.segments]
                    df_data['speaker'] = self.speakers
                
                df = pd.DataFrame(df_data)
                df.to_csv(labels_path, index=False)
                
                # Filtra per split corrente se necessario
                if self.dataset_config.get('has_predefined_splits', False):
                    df_filtered = df[df['split'] == self.split]
                    self.audio_files = [os.path.normpath(filepath.replace('\\', '/')) for filepath in df_filtered['filepath'].tolist()]
                    self.labels = df_filtered['label_id'].tolist()
                    
                    # Carica anche segmentazione se disponibile
                    if self.dataset_config.get('has_segmentation', False) and 'start_ms' in df_filtered.columns:
                        self.segments = list(zip(df_filtered['start_ms'].tolist(), df_filtered['end_ms'].tolist()))
                        self.speakers = df_filtered['speaker'].tolist()
            else:
                print("❌ Preprocessing fallito: nessun file audio trovato nel dataset.")
                raise ValueError(f"Nessun file audio trovato nel dataset {self.dataset_name}")

    def _process_addresso_segmentation(self, split_dir, splits):
        """Processa i file Addresso con segmentazione CSV"""
        audio_dir = os.path.join(split_dir, 'audio')
        segmentation_dir = os.path.join(split_dir, 'segmentation')
        
        if not os.path.exists(audio_dir) or not os.path.exists(segmentation_dir):
            print(f"⚠️ Directory audio o segmentation mancanti in {split_dir}")
            return
        
        # Scansiona le sottocartelle cn e ad
        for class_folder in ['cn', 'ad']:
            class_audio_dir = os.path.join(audio_dir, class_folder)
            class_seg_dir = os.path.join(segmentation_dir, class_folder)
            
            if os.path.exists(class_audio_dir) and os.path.exists(class_seg_dir):
                # Trova tutti i file audio
                for audio_file in os.listdir(class_audio_dir):
                    if audio_file.endswith('.wav'):
                        # Trova il corrispondente file CSV
                        base_name = os.path.splitext(audio_file)[0]
                        csv_file = os.path.join(class_seg_dir, f"{base_name}.csv")
                        
                        if os.path.exists(csv_file):
                            # Leggi segmentazione
                            try:
                                seg_df = pd.read_csv(csv_file)
                                # Filtra solo segmenti PAR (partecipante)
                                par_segments = seg_df[seg_df['speaker'] == 'PAR']
                                
                                audio_path = os.path.join(class_audio_dir, audio_file)
                                audio_path = os.path.normpath(audio_path.replace('\\', '/'))
                                
                                label_id = self.dataset_config['label_dict'][class_folder]
                                
                                # Aggiungi un campione per ogni segmento PAR
                                for _, row in par_segments.iterrows():
                                    start_ms = int(row['begin'] * 1000)  # Converti in ms
                                    end_ms = int(row['end'] * 1000)
                                    
                                    # Filtra segmenti troppo corti (< 0.5 secondi)
                                    if (end_ms - start_ms) >= 500:
                                        self.audio_files.append(audio_path)
                                        self.labels.append(label_id)
                                        self.segments.append((start_ms, end_ms))
                                        self.speakers.append(base_name)  # Usa nome file come speaker ID
                                        splits.append(self.split)
                                        
                            except Exception as e:
                                print(f"⚠️ Errore leggendo {csv_file}: {e}")
        
        print(f"📊 Addresso {self.split}: {len(self.audio_files)} segmenti caricati")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Sample:
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Normalizza il path per compatibilità cross-platform (sicurezza aggiuntiva)
        audio_path = os.path.normpath(audio_path.replace('\\', '/'))
        
        try:
            # Carica audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Validazione: controlla se il waveform è vuoto
            if waveform.numel() == 0:
                print(f"Warning: Empty waveform for file {audio_path}. Skipping...")
                # Ritorna un campione con waveform vuoto che verrà gestito dal preprocessing
                waveform = torch.zeros(1, int(self.cfg['data']['fixed_duration_s'] * self.cfg['data']['target_sr']))
            
            # Se il dataset ha segmentazione, estrai il segmento specifico
            if self.dataset_config.get('has_segmentation', False) and hasattr(self, 'segments') and idx < len(self.segments):
                start_ms, end_ms = self.segments[idx]
                start_sample = int(start_ms * sample_rate / 1000)
                end_sample = int(end_ms * sample_rate / 1000)
                
                # Validazione segmentazione
                if start_sample >= end_sample or start_sample >= waveform.shape[1]:
                    print(f"Warning: Invalid segment [{start_ms}ms-{end_ms}ms] for file {audio_path}. Using full audio.")
                else:
                    waveform = waveform[:, start_sample:end_sample]
                    # Controlla se il segmento risultante è vuoto
                    if waveform.numel() == 0:
                        print(f"Warning: Empty segment for file {audio_path}. Using minimal audio.")
                        waveform = torch.zeros(1, int(0.1 * sample_rate))  # 100ms di silenzio
            
            # Preprocessing unificato per tutti i modelli
            waveform = apply_preprocessing(waveform, sample_rate, self.cfg)
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Crea un waveform di fallback (silenzio)
            target_length = int(self.cfg['data']['fixed_duration_s'] * self.cfg['data']['target_sr'])
            waveform = torch.zeros(1, target_length)
        
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
