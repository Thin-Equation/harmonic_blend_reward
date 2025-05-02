import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def harmonic_blend(sim: float, reward: float, alpha: float = 0.5) -> float:
    """
    Calculate harmonic mean between similarity and reward scores
    
    Args:
        sim: Similarity score between prompt and response
        reward: Reward model score
        alpha: Weight parameter (default: 0.5 for equal weighting)
        
    Returns:
        Harmonic mean of the two scores
    """
    epsilon = 1e-8  # Small value to prevent division by zero
    return 2 * (alpha * sim * (1 - alpha) * reward) / (alpha * sim + (1 - alpha) * reward + epsilon)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

class HarmonicBlendReward:
    """
    Main class that integrates embedding similarity with reward model scoring
    
    This class combines semantic similarity between prompts and responses
    with reward model scores to create a balanced assessment metric.
    """
    
    def __init__(
        self, 
        embedding_model = None,
        reward_model = None,
        alpha: float = 0.5,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the HarmonicBlendReward system
        
        Args:
            embedding_model: Pre-initialized embedding model or None to use default
            reward_model: Pre-initialized reward model or None to use default
            alpha: Balance parameter for harmonic blend (default: 0.5)
            device: Device to use for models ('cuda', 'cpu', etc.)
            cache_dir: Directory for caching model files
        """
        from utils.device_utils import get_optimal_device
        from models.embedding_model import LajavanessEmbedding
        from models.reward_model import QRMRewardModel
        
        # Determine device if not specified
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = device
            
        logger.info(f"Initializing HarmonicBlendReward on device: {self.device}")
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            logger.info("Loading default embedding model")
            self.embedding_model = LajavanessEmbedding(device=self.device, cache_dir=cache_dir)
        else:
            logger.info("Using provided embedding model")
            self.embedding_model = embedding_model
            
        # Initialize reward model if not provided
        if reward_model is None:
            logger.info("Loading default reward model")
            self.reward_model = QRMRewardModel(device=self.device, cache_dir=cache_dir)
        else:
            logger.info("Using provided reward model")
            self.reward_model = reward_model
            
        # Store other parameters
        self.alpha = alpha
        logger.info(f"Blend parameter alpha set to {alpha}")
        
    def score(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Calculate the combined harmonic blend score for a prompt-response pair
        
        Args:
            prompt: User prompt text
            response: Model response text
            
        Returns:
            Dictionary with similarity score, reward score, and harmonic blend score
        """
        # Get embeddings and calculate similarity
        prompt_embedding = self.embedding_model.get_embedding(prompt)
        response_embedding = self.embedding_model.get_embedding(response)
        similarity = cosine_similarity(prompt_embedding, response_embedding)
        
        # Get reward score
        reward = self.reward_model.get_reward_score(prompt, response)
        
        # Calculate harmonic blend
        blend = harmonic_blend(similarity, reward, self.alpha)
        
        # Return all scores for transparency
        return {
            "similarity": float(similarity),
            "reward": float(reward),
            "harmonic_blend": float(blend)
        }
    
    def batch_score(self, prompt_response_pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Score multiple prompt-response pairs
        
        Args:
            prompt_response_pairs: List of (prompt, response) tuples
            
        Returns:
            List of score dictionaries
        """
        results = []
        for prompt, response in prompt_response_pairs:
            results.append(self.score(prompt, response))
        return results
    
    def to(self, device: str):
        """
        Move models to specified device
        
        Args:
            device: Device to move models to ('cuda', 'cpu', etc.)
        """
        logger.info(f"Moving HarmonicBlendReward from {self.device} to {device}")
        self.device = device
        self.embedding_model.to(device)
        self.reward_model.to(device)
        return self
