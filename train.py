import os
import argparse
import yaml
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.utils.data
from matplotlib import pyplot
from matplotlib.axes import Axes
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForAudioClassification
import seaborn as sns
import matplotlib.pyplot as plt

from config.config import OPTIMIZERS, Config
from data_classes.dataset import AudioDataset, Sample
from model_classes.cnn_model import CNNModel
from model_classes.mlp import MLP
from extract_representetion.classical_features import extract_features
from extract_representetion.transformers_features import AudioEmbeddings
from save_model_results import save_model_results


def train_classical_svm(cfg, dataset):
    """Training per SVM classico"""
    print("Training Classical SVM...")
    
    # Estrai features per tutti i campioni con parallelizzazione
    print("🚀 Estraendo features con ottimizzazioni...")
    X, y = [], []
    
    # Usa DataLoader per batch processing delle features
    feature_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    for batch in tqdm(feature_loader, desc="Extracting features (batched)"):
        batch_features = []
        for i in range(len(batch['waveform'])):
            waveform = batch['waveform'][i]
            features = extract_features(waveform, cfg['data']['target_sr'], cfg)
            batch_features.append(features)
        
        X.extend(batch_features)
        y.extend(batch['label'].tolist())
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/validation/test con validation split dal config
    validation_split = cfg['training'].get('validation_split', 0.2)
    test_split = 0.2
    train_split = 1.0 - validation_split - test_split
    
    print(f"📊 Dataset split: Train {train_split:.1%}, Val {validation_split:.1%}, Test {test_split:.1%}")
    
    # Prima divisione: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=cfg['training']['seed'], stratify=y
    )
    
    # Seconda divisione: train vs validation
    val_size_adjusted = validation_split / (1 - test_split)  # Aggiusta per il subset rimanente
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=cfg['training']['seed'], stratify=y_temp
    )
    
    print(f"📊 Training con validation loss monitoring")
    
    # Training SVM
    svm = SVC(
        kernel=cfg['model']['svm']['kernel'],
        C=cfg['model']['svm']['C'],
        gamma=cfg['model']['svm']['gamma'],
        random_state=cfg['training']['seed'],
        probability=True  # Abilita predict_proba per calcolare loss
    )
    
    svm.fit(X_train, y_train)
    
    # Validation loss (usando log-loss per SVM con probabilità)
    from sklearn.metrics import log_loss
    y_val_proba = svm.predict_proba(X_val)
    val_loss = log_loss(y_val, y_val_proba)
    
    print(f"📊 Validation Loss: {val_loss:.4f}")
    
    # Evaluation
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Test Accuracy: {accuracy:.4f}")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Salva tutti i grafici usando save_model_results
    save_model_results(
        model_name="SVM",
        y_true=y_test.tolist(),
        y_pred=y_pred.tolist(),
        dataset_name=cfg['data']['dataset_name'],
        model_config={
            'C': cfg['model']['svm']['C'],
            'kernel': cfg['model']['svm']['kernel'],
            'gamma': cfg['model']['svm']['gamma'],
            'validation_loss': val_loss
        },
        class_names=['healthy', 'parkinson']
    )

    return svm


def train_classical_mlp(cfg, dataset):
    """Training per MLP classico"""
    print("Training Classical MLP...")
    
    device = torch.device(cfg['training']['device'])
    
    # Debug GPU usage
    print(f"🔍 MLP TRAINING DEBUG:")
    print(f"   - Device: {device}")
    if device.type == 'cuda':
        print(f"   - GPU Memory before training: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"   - GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    print(f"{'='*40}")
    
    # Estrai features per tutti i campioni con parallelizzazione
    print("🚀 Estraendo features con ottimizzazioni...")
    X, y = [], []
    
    # Usa DataLoader per batch processing delle features
    feature_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    for batch in tqdm(feature_loader, desc="Extracting features (batched)"):
        batch_features = []
        for i in range(len(batch['waveform'])):
            waveform = batch['waveform'][i]
            features = extract_features(waveform, cfg['data']['target_sr'], cfg)
            batch_features.append(features)
        
        X.extend(batch_features)
        y.extend(batch['label'].tolist())
    
    X = np.array(X)
    y = np.array(y)
    
    # Converti a tensori
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split train/validation/test
    indices = torch.randperm(len(X_tensor))
    
    # Usa validation_split dalla configurazione
    val_split = cfg['training'].get('validation_split', 0.2)
    test_split = 0.2  # Mantieni 20% per test
    train_split = 1.0 - val_split - test_split
    
    train_size = int(train_split * len(indices))
    val_size = int(val_split * len(indices))
    test_size = len(indices) - train_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train, X_val, X_test = X_tensor[train_indices], X_tensor[val_indices], X_tensor[test_indices]
    y_train, y_val, y_test = y_tensor[train_indices], y_tensor[val_indices], y_tensor[test_indices]
    
    print(f"📊 Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Crea dataset e dataloader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    # Crea modello
    num_classes = len(set(y))
    input_dim = X.shape[1]
    model = MLP(input_dim, num_classes, cfg).to(device)
    
    # Debug model on GPU
    print(f"🎯 MODEL DEBUG:")
    print(f"   - Model device: {next(model.parameters()).device}")
    print(f"   - Model parameters count: {sum(p.numel() for p in model.parameters())}")
    if device.type == 'cuda':
        print(f"   - GPU Memory after model creation: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"{'='*40}")
    
    # Training setup
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training'].get('max_lr', 0.001))
    
    print(f"📊 Training con validation loss monitoring")
    
    # Training loop con validation loss
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(cfg['training']['epochs']):
        # Training phase
        total_train_loss = 0
        model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Debug first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"🔍 FIRST BATCH DEBUG:")
                print(f"   - Batch input device: {batch_x.device}")
                print(f"   - Batch target device: {batch_y.device}")
                if device.type == 'cuda':
                    print(f"   - GPU Memory during training: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
                print(f"{'='*40}")
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss = criterion(outputs, batch_y)
                total_val_loss += val_loss.item()
                
                # Calcola accuracy di validation
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            if device.type == 'cuda':
                print(f"   - GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Raccogli predizioni e etichette per il classification report
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = correct / total
    print(f"MLP Test Accuracy: {accuracy:.4f}")
    
    # Stampa classification report
    print("\nMLP Classification Report:")
    print(classification_report(all_labels, all_predictions))
    
    # Salva tutti i grafici usando save_model_results
    save_model_results(
        model_name="MLP",
        y_true=all_labels,
        y_pred=all_predictions,
        dataset_name=cfg['data']['dataset_name'],
        model_config={
            'hidden_layers': cfg['model']['mlp']['hidden'],
            'dropout': cfg['model']['mlp']['dropout'],
            'learning_rate': cfg['training']['max_lr'],
            'epochs': cfg['training']['epochs'],
            'batch_size': cfg['training']['batch_size']
        },
        class_names=['healthy', 'parkinson']
    )

    return model


def train_transformers_mlp(cfg, dataset):
    """Training per Transformers + MLP"""
    print("Training Transformers + MLP...")
    
    device = torch.device(cfg['training']['device'])
    
    # Debug GPU usage for transformers
    print(f"🔍 TRANSFORMERS+MLP DEBUG:")
    print(f"   - Device: {device}")
    if device.type == 'cuda':
        print(f"   - GPU Memory before model loading: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"{'='*40}")
    
    # Inizializza estrattore features transformers
    model_name = cfg['features']['transformers']['model_name']
    sampling_rate = cfg['data']['target_sr']
    feature_extractor = AudioEmbeddings(model_name, device, sampling_rate)

    print("Model:", model_name)
    
    # Debug after transformer model loading
    if device.type == 'cuda':
        print(f"🎯 After transformer loading - GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    # Crea dataloader con validation split
    val_split = cfg['training'].get('validation_split', 0.2)
    test_split = 0.2  # Mantieni 20% per test
    train_split = 1.0 - val_split - test_split
    
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"📊 Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # Determina dimensione embedding
    sample_batch = next(iter(train_loader))
    sample_waveform = sample_batch['waveform'][0].squeeze().numpy()  # Prendi il primo sample del batch
    sample_embeddings = feature_extractor.extract(sample_waveform)
    embedding_dim = sample_embeddings.shape[1]
    
    # Crea MLP head
    num_classes = 2  # healthy (0) e parkinson (1)
    mlp_head = MLP(embedding_dim, num_classes, cfg).to(device)
    
    # Training setup
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_head.parameters(), lr=cfg['training'].get('max_lr', 0.001))
    
    print(f"📊 Training con validation loss monitoring")
    
    # Training loop con validation loss
    train_losses = []
    val_losses = []
    
    mlp_head.train()
    
    for epoch in tqdm(range(cfg['training']['epochs']), desc="Training Transformers+MLP"):
        # Training phase
        total_train_loss = 0
        mlp_head.train()
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}", leave=False)
        for batch in epoch_pbar:
            waveforms = batch['waveform']
            labels = batch['label'].long().to(device)  # Assicura che le labels siano LongTensor
            
            # Estrai embeddings per ogni waveform nel batch
            batch_embeddings = []
            for waveform in waveforms:
                waveform_np = waveform.squeeze().numpy()
                embeddings = feature_extractor.extract(waveform_np)
                batch_embeddings.append(embeddings)
            
            # Stack degli embeddings in un batch tensor
            embeddings = torch.stack(batch_embeddings).to(device)
            # Rimuovi la dimensione extra: da [batch, 1, features] a [batch, features]
            embeddings = embeddings.squeeze(1)
            
            # Forward MLP
            optimizer.zero_grad()
            logits = mlp_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            epoch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        mlp_head.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch['waveform']
                labels = batch['label'].long().to(device)
                
                # Estrai embeddings per ogni waveform nel batch
                batch_embeddings = []
                for waveform in waveforms:
                    waveform_np = waveform.squeeze().numpy()
                    embeddings = feature_extractor.extract(waveform_np)
                    batch_embeddings.append(embeddings)
                
                # Stack degli embeddings in un batch tensor
                embeddings = torch.stack(batch_embeddings).to(device)
                embeddings = embeddings.squeeze(1)
                
                # Forward MLP
                logits = mlp_head(embeddings)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                
                # Calcola accuracy di validation
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Evaluation
    mlp_head.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch['waveform']
            labels = batch['label'].long().to(device)  # Assicura che le labels siano LongTensor
            
            # Estrai embeddings per ogni waveform nel batch
            batch_embeddings = []
            for waveform in waveforms:
                waveform_np = waveform.squeeze().numpy()
                embeddings = feature_extractor.extract(waveform_np)
                batch_embeddings.append(embeddings)
            
            # Stack degli embeddings in un batch tensor
            embeddings = torch.stack(batch_embeddings).to(device)
            # Rimuovi la dimensione extra: da [batch, 1, features] a [batch, features]
            embeddings = embeddings.squeeze(1)
            
            logits = mlp_head(embeddings)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Raccogli predizioni e etichette per il classification report
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    print(f"Transformers+MLP Test Accuracy: {accuracy:.4f}")
    
    # Stampa classification report
    print("\nTransformers+MLP Classification Report:")
    print(classification_report(all_labels, all_predictions))
    
    # Salva tutti i grafici usando save_model_results
    save_model_results(
        model_name=f"Transformers+MLP_{model_name.replace('/', '_')}",
        y_true=all_labels,
        y_pred=all_predictions,
        dataset_name=cfg['data']['dataset_name'],
        model_config={
            'transformer_model': cfg['features']['transformers']['model_name'],
            'hidden_layers': cfg['model']['mlp']['hidden'],
            'dropout': cfg['model']['mlp']['dropout'],
            'learning_rate': cfg['training']['max_lr'],
            'epochs': cfg['training']['epochs'],
            'batch_size': cfg['training']['batch_size'],
            'device': cfg['training']['device']
        },
        class_names=['healthy', 'parkinson']
    )
    
    return mlp_head


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/base_config.yaml', help='Config file path')
    parser.add_argument('--data.dataset_name', dest='dataset_name', help='Dataset name override')
    parser.add_argument('--model.branch', dest='branch', help='Model branch override')
    parser.add_argument('--model.cnn.in_type', dest='cnn_in_type', help='CNN input type override')
    parser.add_argument('--features.transformers.model_name', dest='transformer_model', help='Transformer model override')
    args = parser.parse_args()
    
    # Carica configurazione
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override da CLI
    if args.dataset_name:
        cfg['data']['dataset_name'] = args.dataset_name
    if args.branch:
        cfg['model']['branch'] = args.branch
    if args.cnn_in_type:
        cfg['model']['cnn']['in_type'] = args.cnn_in_type
    if args.transformer_model:
        cfg['features']['transformers']['model_name'] = args.transformer_model
    
    # Imposta seed
    torch.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])
    
    print(f"Training with branch: {cfg['model']['branch']}")
    print(f"Dataset: {cfg['data']['dataset_name']}")
    
    # Carica dataset
    dataset = AudioDataset(cfg)
    
    # Switch sui rami
    branch = cfg['model']['branch']
    
    if branch == 'classical_svm':
        model = train_classical_svm(cfg, dataset)
    elif branch == 'classical_mlp':
        model = train_classical_mlp(cfg, dataset)
    elif branch == 'cnn':
        # Training CNN (adattato per nuova configurazione)
        print("Training CNN...")
        
        device = torch.device(cfg['training']['device'])
        
        # Split dataset con validation split dal config
        validation_split = cfg['training'].get('validation_split', 0.2)
        test_split = 0.2
        train_split = 1.0 - validation_split - test_split
        
        print(f"📊 Dataset split: Train {train_split:.1%}, Val {validation_split:.1%}, Test {test_split:.1%}")
        
        # Prima divisione: train+val vs test
        temp_size = int((1 - test_split) * len(dataset))
        test_size = len(dataset) - temp_size
        temp_dataset, test_dataset = torch.utils.data.random_split(dataset, [temp_size, test_size])
        
        # Seconda divisione: train vs validation
        val_size = int(validation_split * len(dataset))
        train_size = temp_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, [train_size, val_size])
        
        # Dataloader
        # DataLoader ottimizzati per massima velocità
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg['training']['batch_size'], 
            shuffle=True,
            num_workers=4,  # Parallelizzazione caricamento dati
            pin_memory=True,  # Trasferimento GPU più veloce
            persistent_workers=True  # Riutilizzo workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg['training']['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg['training']['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Determina input size per CNN
        sample_batch = next(iter(train_loader))
        sample_waveform = sample_batch['waveform']
        
        if cfg['model']['cnn']['in_type'] == 'spectrogram':
            # Input è spettrogramma [batch, n_mels, time]
            input_size = sample_waveform.shape[-1]  # time dimension
        else:
            # Input è waveform [batch, 1, time]
            input_size = sample_waveform.shape[-1]  # time dimension
        
        # Crea modello CNN (riusa quello esistente)
        model = CNNModel(waveform_size=input_size, dropout=cfg['model']['dropout'], device=device)
        model.to(device)
        
        # Training setup
        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training'].get('max_lr', 0.001))
        
        print(f"📊 Training con validation loss monitoring")
        
        # Training loop con validation loss
        train_losses = []
        val_losses = []
        
        for epoch in tqdm(range(cfg['training']['epochs']), desc="Training CNN"):
            # Training phase
            model.train()
            total_train_loss = 0
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}", leave=False)
            for batch in epoch_pbar:
                waveforms = batch['waveform'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                epoch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase - solo ogni 3 epoche per velocità
            if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == cfg['training']['epochs'] - 1:
                model.eval()
                total_val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        waveforms = batch['waveform'].to(device, non_blocking=True)
                        labels = batch['label'].to(device, non_blocking=True)
                        
                        outputs = model(waveforms)
                        val_loss = criterion(outputs, labels)
                        
                        total_val_loss += val_loss.item()
                        
                        # Calcola accuratezza
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Train Loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                outputs = model(waveforms)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Raccogli predizioni e etichette per il classification report
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        print(f"CNN Test Accuracy: {accuracy:.4f}")
        
        # Stampa classification report
        print("\nCNN Classification Report:")
        print(classification_report(all_labels, all_predictions))
        
        # Salva tutti i grafici usando save_model_results
        save_model_results(
            model_name="CNN",
            y_true=all_labels,
            y_pred=all_predictions,
            dataset_name=cfg['data']['dataset_name'],
            model_config={
                'in_type': cfg['model']['cnn']['in_type'],
                'dropout': cfg['model']['dropout'],
                'learning_rate': cfg['training']['max_lr'],
                'epochs': cfg['training']['epochs'],
                'batch_size': cfg['training']['batch_size'],
                'device': cfg['training']['device']
            },
            class_names=['healthy', 'parkinson']
        )

        # Salva il modello
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/cnn_model.pt')
        
    elif branch == 'transformers_mlp':
        model = train_transformers_mlp(cfg, dataset)
        
        # Salva modelli
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/transformers_mlp_head.pt')
        
    else:
        raise ValueError(f"Branch non supportato: {branch}")
    
    print("Training completato!")
