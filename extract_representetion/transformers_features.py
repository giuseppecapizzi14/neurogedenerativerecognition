from transformers import AutoFeatureExtractor, AutoModel
import torch


class AudioEmbeddings:
    '''
    This class is intended to extract embeddings from audio models.
    It uses Wav2Vec2 as a default model.
    '''
    
    def __init__(self, model_name, device, sampling_rate=16000):
        
        self.processor = AutoFeatureExtractor.from_pretrained(model_name, sampling_rate=sampling_rate)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = device
        self.model.to(self.device)
        
        self.model_name = model_name
        self.sampling_rate = sampling_rate  # Salva sampling_rate per usarlo nel metodo extract
        
        # eval mode
        self.model.eval()
        
    def extract(self, speech):
        '''
        Extract embeddings from a speech.
        
        Args:
            speech: Speech to extract embeddings from.
        
        Returns:
            torch.Tensor: Embeddings of the speech.
        '''
        
        inputs = self.processor(speech, return_tensors="pt", padding="longest", sampling_rate=self.sampling_rate)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1)
    
    def extract_batch(self, speech_batch):
        '''
        Extract embeddings from a batch of speeches - MUCH FASTER!
        
        Args:
            speech_batch: List of speeches to extract embeddings from.
        
        Returns:
            torch.Tensor: Batch of embeddings [batch_size, embedding_dim].
        '''
        
        # Process entire batch at once - this is the key optimization!
        inputs = self.processor(speech_batch, return_tensors="pt", padding="longest", sampling_rate=self.sampling_rate)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Return mean pooled embeddings for each sample in batch
        return outputs.last_hidden_state.mean(dim=1)  # [batch_size, embedding_dim]