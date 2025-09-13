import torch
from torch import Tensor
from transformers import AutoModelForAudioClassification
from typing import Dict, Any


class TransformersFeatureExtractor:
    """
    Classe per estrarre features dai modelli transformers pre-addestrati.
    Gestisce il caricamento del backbone e l'estrazione degli embeddings.
    """
    
    def __init__(self, model_name: str, device: torch.device, freeze_backbone: bool = True):
        """
        Inizializza l'estrattore di features transformers.
        
        Args:
            model_name: Nome del modello transformers da utilizzare
            device: Device su cui caricare il modello
            freeze_backbone: Se congelare i parametri del backbone
        """
        self.model_name = model_name
        self.device = device
        self.freeze_backbone = freeze_backbone
        
        # Carica backbone transformers
        self.backbone = AutoModelForAudioClassification.from_pretrained(
            model_name, 
            output_hidden_states=True,
            num_labels=2  # placeholder
        ).to(device)
        
        # Congela backbone se richiesto
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.backbone.eval()
    
    def extract_embeddings(self, hf_inputs: Dict[str, Tensor], pooling: str = 'mean') -> Tensor:
        """
        Estrae embeddings dal backbone transformers.
        
        Args:
            hf_inputs: Input preprocessati per il modello transformers
            pooling: Metodo di pooling ('mean' o 'cls')
            
        Returns:
            Tensor: Embeddings estratti [batch_size, embedding_dim]
        """
        with torch.no_grad():
            outputs = self.backbone(**hf_inputs)
            hidden_states = outputs.hidden_states[-1]  # ultimo layer
            
            if pooling == 'mean':
                embeddings = hidden_states.mean(dim=1)
            else:  # cls
                embeddings = hidden_states[:, 0]
                
        return embeddings
    
    def get_embedding_dim(self, sample_inputs: Dict[str, Tensor], pooling: str = 'mean') -> int:
        """
        Determina la dimensione degli embeddings.
        
        Args:
            sample_inputs: Input di esempio per determinare la dimensione
            pooling: Metodo di pooling utilizzato
            
        Returns:
            int: Dimensione degli embeddings
        """
        with torch.no_grad():
            outputs = self.backbone(**sample_inputs)
            hidden_states = outputs.hidden_states[-1]
            
            if pooling == 'mean':
                return hidden_states.mean(dim=1).shape[1]
            else:  # cls
                return hidden_states[:, 0].shape[1]