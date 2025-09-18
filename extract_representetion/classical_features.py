import numpy as np
import librosa
import parselmouth
from typing import Dict, Any
from torch import Tensor
from sklearn.preprocessing import StandardScaler


def extract_features(waveform: Tensor, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Estrae features classiche da un waveform audio.
    
    Args:
        waveform: Tensor audio [1, T]
        sr: Sample rate
        cfg: Configurazione
    
    Returns:
        np.ndarray: Vettore 1D di features normalizzate
    """
    # Converti a numpy
    audio = waveform.squeeze().numpy()
    
    features = []
    
    # MFCC features
    n_mfcc = cfg['features']['classical']['n_mfcc']
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Statistiche MFCC (mean, std)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # Delta MFCC (se richiesto)
    if cfg['features']['classical']['mfcc_delta']:
        delta_mfccs = librosa.feature.delta(mfccs)
        features.extend(np.mean(delta_mfccs, axis=1))
        features.extend(np.std(delta_mfccs, axis=1))
    
    # Delta-Delta MFCC (se richiesto)
    if cfg['features']['classical'].get('mfcc_delta2', False):
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features.extend(np.mean(delta2_mfccs, axis=1))
        features.extend(np.std(delta2_mfccs, axis=1))
    
    # Energia (RMS)
    if cfg['features']['classical']['energy']:
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(rms), np.std(rms)])
    
    # Pitch features
    if cfg['features']['classical']['pitch']:
        pitch_method = cfg['features']['classical'].get('pitch_method', 'pyin')
        pitch_fmin = cfg['features']['classical'].get('pitch_fmin', 50)
        pitch_fmax = cfg['features']['classical'].get('pitch_fmax', 400)
        
        if pitch_method == 'pyin':
            # Usa librosa per pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=pitch_fmin, 
                fmax=pitch_fmax, 
                sr=sr
            )
            
            # Rimuovi valori NaN
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                features.extend([
                    np.mean(f0_clean),
                    np.std(f0_clean),
                    np.min(f0_clean),
                    np.max(f0_clean),
                    np.median(f0_clean),
                    len(f0_clean) / len(f0)  # Voicing ratio
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
    
    # Jitter e Shimmer usando Parselmouth
    if cfg['features']['classical']['jitter_shimmer']:
        try:
            # Crea oggetto Sound per Parselmouth
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Calcola jitter e shimmer
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
            
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Gestisci valori undefined
            jitter = jitter if not np.isnan(jitter) and not np.isinf(jitter) else 0
            shimmer = shimmer if not np.isnan(shimmer) and not np.isinf(shimmer) else 0
            
            features.extend([jitter, shimmer])
            
        except Exception:
            # In caso di errore, aggiungi valori zero
            features.extend([0, 0])
    
    # Features spettrali aggiuntive
    if cfg['features']['classical'].get('spectral_features', False):
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend(np.mean(spectral_contrast, axis=1))
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        features.extend([np.mean(spectral_flatness), np.std(spectral_flatness)])
    
    # Zero crossing rate
    if cfg['features']['classical'].get('zero_crossing_rate', False):
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([np.mean(zcr), np.std(zcr)])
    
    # Converti a array e gestisci valori NaN/Inf
    features_array = np.array(features, dtype=np.float32)
    
    # Sostituisci NaN e Inf con 0
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_array