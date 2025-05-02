#!/usr/bin/env python
"""
Example script for Harmonic Blend Reward

This example demonstrates how to use the Harmonic Blend Reward system
to evaluate LLM responses using a combination of semantic similarity
and reward model scoring.
"""

import logging
import os
from models.core import HarmonicBlendReward
from utils.device_utils import get_optimal_device
import visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    # Set device
    device = get_optimal_device()
    
    # Initialize the Harmonic Blend Reward model
    logger.info("Initializing Harmonic Blend Reward model")
    model = HarmonicBlendReward(
        device=device,
        alpha=0.6  # Giving slightly more weight to reward model
    )
    
    # Example prompts and responses for evaluation
    examples = [
        (
            "Explain how solar panels work",
            "Solar panels work by using photovoltaic cells to convert sunlight into electricity. When photons from sunlight hit the semiconductor material in the cells, they knock electrons loose, creating an electric current. This direct current (DC) is then converted to alternating current (AC) by an inverter for use in homes and businesses."
        ),
        (
            "Explain how solar panels work",
            "Solar panels use the photovoltaic effect to generate electricity from sunlight."
        ),
        (
            "Write a poem about the ocean",
            "Endless blue stretches far and wide,\nWaves dance with the rhythm of the tide.\nSunlight sparkles on waters deep,\nSecrets the ocean will forever keep.\n\nMighty and ancient, powerful and free,\nLife flourishes in the boundless sea.\nFrom shore to horizon, a magnificent sight,\nThe ocean's vastness fills me with delight."
        ),
        (
            "How do I bake chocolate chip cookies?",
            "To make chocolate chip cookies, you need butter, sugar, eggs, vanilla, flour, baking soda, salt and chocolate chips. First, cream together butter and sugars, then add eggs and vanilla. Mix in dry ingredients, fold in chocolate chips. Drop spoonfuls onto baking sheets and bake at 375°F for about 10 minutes or until golden brown."
        ),
        (
            "How do I bake chocolate chip cookies?",
            "Sorry, I don't have that information."
        )
    ]
    
    # Score each example
    logger.info("Scoring examples")
    results = []
    
    for i, (prompt, response) in enumerate(examples):
        logger.info(f"Example {i+1}:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")
        
        # Get scores
        scores = model.score(prompt, response)
        results.append(scores)
        
        logger.info(f"Similarity: {scores['similarity']:.4f}")
        logger.info(f"Reward: {scores['reward']:.4f}")
        logger.info(f"Harmonic Blend: {scores['harmonic_blend']:.4f}")
        logger.info("-" * 50)
    
    # Convert to DataFrame and visualize
    try:
        import pandas as pd
        from evaluation import RewardEvaluator
        
        # Create results DataFrame
        df = pd.DataFrame({
            'prompt': [p for p, _ in examples],
            'response': [r for _, r in examples],
            **{k: [s[k] for s in results] for k in results[0].keys()}
        })
        
        # Create evaluator
        evaluator = RewardEvaluator(results)
        
        # Get summary statistics
        stats = evaluator.summary_stats()
        logger.info("Summary Statistics:")
        for metric, values in stats.items():
            logger.info(f"{metric}: mean={values['mean']:.4f}, median={values['median']:.4f}")
        
        # Create output directory
        os.makedirs('./results', exist_ok=True)
        
        # Generate visualization report
        logger.info("Creating visualization report")
        visualization.create_report(df, evaluator, output_dir='./results')
        
    except ImportError as e:
        logger.warning(f"Couldn't generate visualizations: {e}")
    
    logger.info("Finished example")


if __name__ == "__main__":
    main()