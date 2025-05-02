"""
Harmonic Blend Reward Model package.

This package provides a combined approach for evaluating text quality using:
1. Semantic similarity between prompts and responses
2. LLM-based reward modeling
"""

from .core import HarmonicBlendReward, harmonic_blend, cosine_similarity
from .reward_model import QRMRewardModel
from .embedding_model import LajavanessEmbedding

__all__ = [
    'HarmonicBlendReward',
    'QRMRewardModel',
    'LajavanessEmbedding',
    'harmonic_blend',
    'cosine_similarity'
]