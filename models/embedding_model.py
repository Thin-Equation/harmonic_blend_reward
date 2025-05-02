import torch
import numpy as np
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LajavanessEmbedding:
    """Text embedding model for similarity calculations"""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Always use float32 for all devices to ensure compatibility
        self.dtype = torch.float32
        logger.info(f"Using torch.float32 for {self.device} to ensure compatibility")

        logger.info(f"Loading embedding model {model_name_or_path} on {self.device}")

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir,
                torch_dtype=self.dtype
            )
            
            # Explicitly ensure model is in float32
            self.model = self.model.to(dtype=torch.float32)
            
            # Move to device after setting dtype
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Check for NVIDIA GPUs
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                logger.info(f"Running embedding model on GPU: {gpu_name}")
                
                # Check specifically for GH200 or other Hopper architecture
                if "gh200" in gpu_name or "hopper" in gpu_name or "h100" in gpu_name:
                    logger.info("Detected Grace Hopper architecture for embedding model")

            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def to(self, device):
        """Move model to the specified device"""
        logger.info(f"Moving embedding model from {self.device} to {device}")
        self.device = device
        
        # Always keep float32 dtype when moving to any device
        if hasattr(self, 'model'):
            self.model = self.model.to(dtype=torch.float32, device=device)
            
        return self

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings using consistent dtype
            with torch.no_grad():
                # Never use autocast - always use float32 for consistency
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)

            # Normalize and return as numpy array
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return normalized_embeddings[0].cpu().numpy()

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            logger.error(f"Device: {self.device}, Model dtype: {next(self.model.parameters()).dtype}")
            # Return zeros array as fallback
            return np.zeros(1024)  # Assuming 1024-dim embeddings

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling operation to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
