import numpy as np
import librosa
import parselmouth
from typing import Dict, Any
from torch import Tensor


def extract_features(waveform: Tensor, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Estrae features classiche da un waveform audio.
    
    Args:
        waveform: Tensor audio [1, T]
        sr: Sample rate
        cfg: Configurazione
    
    Returns:
        np.ndarray: Vettore 1D di features
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
    
    # Energia (RMS)
    if cfg['features']['classical']['energy']:
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(rms), np.std(rms)])
    
    # Pitch features
    if cfg['features']['classical']['pitch']:
        pitch_cfg = cfg['features']['classical']['pitch']
        
        if pitch_cfg['method'] == 'pyin':
            # Usa librosa per pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=pitch_cfg['fmin'], 
                fmax=pitch_cfg['fmax'], 
                sr=sr
            )
            
            # Rimuovi valori NaN
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                features.extend([
                    np.mean(f0_clean),
                    np.std(f0_clean),
                    np.min(f0_clean),
                    np.max(f0_clean)
                ])
            else:
                features.extend([0, 0, 0, 0])
    
    # Jitter e Shimmer usando Parselmouth
    if cfg['features']['classical']['jitter_shimmer']:
        try:
            # Crea oggetto Sound per Parselmouth
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Estrai pitch
            pitch = sound.to_pitch()
            
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
    
    return np.array(features, dtype=np.float32)