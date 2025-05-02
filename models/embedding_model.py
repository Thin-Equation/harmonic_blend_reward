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
        from utils.device_utils import get_torch_dtype_for_device
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Get optimal dtype for the device
        self.dtype = get_torch_dtype_for_device(torch.device(self.device))
        logger.info(f"Using dtype: {self.dtype} for device: {self.device}")

        logger.info(f"Loading embedding model {model_name_or_path} on {self.device}")

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir,
                torch_dtype=self.dtype
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Apply optimizations for A100 if available
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "a100" in gpu_name:
                    logger.info("Applying optimizations for A100 GPU")
                    # Enable memory efficient attention if available (for large embeddings)
                    if hasattr(self.model.config, "use_flash_attention_2"):
                        self.model.config.use_flash_attention_2 = True
                        logger.info("Enabled Flash Attention 2 for embedding model")

            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def to(self, device):
        """Move model to the specified device"""
        logger.info(f"Moving embedding model from {self.device} to {device}")
        self.device = device
        self.model = self.model.to(device)
        
        # Update dtype based on new device
        from utils.device_utils import get_torch_dtype_for_device
        self.dtype = get_torch_dtype_for_device(torch.device(self.device))
        
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

            # Generate embeddings using the appropriate dtype
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=True):
                    outputs = self.model(**inputs)
                    # Use mean pooling for sentence embedding
                    attention_mask = inputs["attention_mask"]
                    embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)

            # Normalize and return as numpy array
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return normalized_embeddings[0].cpu().numpy()

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zeros array as fallback
            return np.zeros(1024)  # Assuming 1024-dim embeddings

    @property
    def device_type(self):
        """Get the device type string for autocast"""
        if self.device == "cuda":
            return "cuda"
        elif self.device == "mps":
            return "mps"
        else:
            return "cpu"

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling operation to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
