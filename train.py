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
    
    # Estrai features per tutti i campioni
    X, y = [], []
    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        sample = dataset[i]
        waveform = sample['waveform']
        label = sample['label']
        
        features = extract_features(waveform, cfg['data']['target_sr'], cfg)
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg['training']['seed'], stratify=y
    )
    
    # Training SVM
    svm = SVC(
        kernel=cfg['model']['svm']['kernel'],
        C=cfg['model']['svm']['C'],
        gamma=cfg['model']['svm']['gamma'],
        random_state=cfg['training']['seed']
    )
    
    svm.fit(X_train, y_train)
    
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
            'gamma': cfg['model']['svm']['gamma']
        },
        class_names=['healthy', 'parkinson']
    )

    return svm


def train_classical_mlp(cfg, dataset):
    """Training per MLP classico"""
    print("Training Classical MLP...")
    
    device = torch.device(cfg['training']['device'])
    
    # Estrai features per tutti i campioni
    X, y = [], []
    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        sample = dataset[i]
        waveform = sample['waveform']
        label = sample['label']
        
        features = extract_features(waveform, cfg['data']['target_sr'], cfg)
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Converti a tensori
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split train/test
    indices = torch.randperm(len(X_tensor))
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, X_test = X_tensor[train_indices], X_tensor[test_indices]
    y_train, y_test = y_tensor[train_indices], y_tensor[test_indices]
    
    # Crea dataset e dataloader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # Crea modello
    num_classes = len(set(y))
    input_dim = X.shape[1]
    model = MLP(input_dim, num_classes, cfg).to(device)
    
    # Training setup
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training'].get('max_lr', 0.001))
    
    # Training loop
    model.train()
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
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
    
    # Inizializza estrattore features transformers
    model_name = cfg['features']['transformers']['model_name']
    sampling_rate = cfg['data']['target_sr']
    feature_extractor = AudioEmbeddings(model_name, device, sampling_rate)

    print("Model:", model_name)
    
    # Crea dataloader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
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
    
    # Training loop
    mlp_head.train()
    
    for epoch in tqdm(range(cfg['training']['epochs']), desc="Training Transformers+MLP"):
        total_loss = 0
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
            
            total_loss += loss.item()
            epoch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
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
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Dataloader
        train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
        
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
        
        # Training loop
        model.train()
        for epoch in tqdm(range(cfg['training']['epochs']), desc="Training CNN"):
            total_loss = 0
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}", leave=False)
            for batch in epoch_pbar:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
        
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
