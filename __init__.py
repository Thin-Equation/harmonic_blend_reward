"""
Harmonic Blend Reward

A library for evaluating LLM responses using a blend of semantic similarity and reward model scoring.
"""

from models.core import HarmonicBlendReward
from models.reward_model import QRMRewardModel
from models.embedding_model import LajavanessEmbedding

# Export evaluation utilities
from evaluation import RewardEvaluator, evaluate_reward_model

# Version information
__version__ = "0.1.0"

__all__ = [
    'HarmonicBlendReward',
    'QRMRewardModel', 
    'LajavanessEmbedding',
    'RewardEvaluator',
    'evaluate_reward_model',
]