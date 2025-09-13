import os
import argparse
import yaml
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

from config.config import OPTIMIZERS, Config
from data_classes.dataset import AudioDataset, Sample
from metrics import Metrics, MetricsHistory, compute_metrics, evaluate, print_metrics
from model_classes.cnn_model import CNNModel
from model_classes.mlp import MLP
from extract_representetion.classical_features import extract_features
from extract_representetion.transformers_features import TransformersFeatureExtractor


def train_one_epoch(
    model: Module,
    dataloader: DataLoader[Sample],
    loss_criterion: CrossEntropyLoss,
    scheduler: LRScheduler,
    device: torch.device
) -> Metrics:
    model.train()
    total_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    for batch in tqdm(dataloader, desc = "Training"):
        waveforms: Tensor = batch["waveform"]
        waveforms = waveforms.to(device)

        labels: Tensor = batch["label"]
        labels = labels.to(device)

        scheduler.optimizer.zero_grad()

        outputs: Tensor = model(waveforms)
        loss: Tensor = loss_criterion(outputs, labels)
        loss.backward() # type: ignore
        total_loss += loss.item()

        scheduler.optimizer.step()
        scheduler.step()

        pred = torch.argmax(outputs, dim = 1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())

    return compute_metrics(predictions, references, total_loss, len(dataloader))

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
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
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
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    print(f"MLP Test Accuracy: {accuracy:.4f}")
    
    return model


def train_transformers_mlp(cfg, dataset):
    """Training per Transformers + MLP"""
    print("Training Transformers + MLP...")
    
    device = torch.device(cfg['training']['device'])
    
    # Inizializza estrattore features transformers
    model_name = cfg['features']['transformers']['model_name']
    freeze_backbone = cfg['features']['transformers']['freeze_backbone']
    feature_extractor = TransformersFeatureExtractor(model_name, device, freeze_backbone)
    
    # Crea dataloader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # Determina dimensione embedding
    sample_batch = next(iter(train_loader))
    sample_inputs = sample_batch['hf_inputs']
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    pooling = cfg['features']['transformers']['pooling']
    embedding_dim = feature_extractor.get_embedding_dim(sample_inputs, pooling)
    
    # Crea MLP head
    num_classes = len(set([dataset[i]['label'] for i in range(min(100, len(dataset)))]))
    mlp_head = MLP(embedding_dim, num_classes, cfg).to(device)
    
    # Training setup
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_head.parameters(), lr=cfg['training'].get('max_lr', 0.001))
    
    # Training loop
    mlp_head.train()
    
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch in train_loader:
            hf_inputs = batch['hf_inputs']
            labels = batch['label'].to(device)
            
            hf_inputs = {k: v.to(device) for k, v in hf_inputs.items()}
            
            # Estrai embeddings
            embeddings = feature_extractor.extract_embeddings(hf_inputs, pooling)
            
            # Forward MLP
            optimizer.zero_grad()
            logits = mlp_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    mlp_head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            hf_inputs = batch['hf_inputs']
            labels = batch['label'].to(device)
            
            hf_inputs = {k: v.to(device) for k, v in hf_inputs.items()}
            
            embeddings = feature_extractor.extract_embeddings(hf_inputs, pooling)
            logits = mlp_head(embeddings)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Transformers+MLP Test Accuracy: {accuracy:.4f}")
    
    return feature_extractor.backbone, mlp_head


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
        for epoch in range(cfg['training']['epochs']):
            total_loss = 0
            for batch in train_loader:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                outputs = model(waveforms)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"CNN Test Accuracy: {accuracy:.4f}")
        
        # Salva modello
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/cnn_model.pt')
        
    elif branch == 'transformers_mlp':
        model = train_transformers_mlp(cfg, dataset)
        
        # Salva modelli
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model[1].state_dict(), 'checkpoints/transformers_mlp_head.pt')
        
    else:
        raise ValueError(f"Branch non supportato: {branch}")
    
    print("Training completato!")
