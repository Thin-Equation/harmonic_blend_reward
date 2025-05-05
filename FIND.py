#!/usr/bin/env python
"""
Find the optimal alpha parameter for HarmonicBlendReward on RewardBench.

This script evaluates the HarmonicBlendReward model with different alpha values
to find the optimal balance between similarity and reward scores.
"""

import os
import logging
import argparse
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from models.core import HarmonicBlendReward
from utils.device_utils import get_optimal_device, free_memory
from datasets import load_dataset

# Import functions from evaluate_rewardbench.py
from evaluate_rewardbench import compute_pairwise_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Replicate the load_rewardbench function here to make this script more self-contained
def load_rewardbench(split: str = "filtered", limit: int = None):
    """
    Load the RewardBench dataset
    
    Args:
        split: Dataset split to use ("filtered" or "raw")
        limit: Maximum number of examples to load (None for all)
        
    Returns:
        Tuple of (prompts, responses, labels)
    """
    logger.info(f"Loading RewardBench dataset (split: {split})")
    
    # Load dataset
    dataset = load_dataset("allenai/reward-bench")
    
    # Get the appropriate split
    if split not in ["filtered", "raw"]:
        logger.warning(f"Split '{split}' not recognized, using 'filtered' instead")
        split = "filtered"
        
    subset = dataset[split]
        
    # Extract data
    prompts = []
    response_lists = []
    label_lists = []
    
    for i, item in enumerate(subset):
        if limit and i >= limit:
            break
            
        prompts.append(item["prompt"])
        
        # RewardBench has pairs of chosen/rejected responses
        # Format: each prompt has a list of responses and corresponding labels
        # Label 1 for chosen (preferred) and 0 for rejected (less preferred)
        responses = [item["chosen"], item["rejected"]]
        labels = [1, 0]  # 1 for chosen, 0 for rejected
        
        response_lists.append(responses)
        label_lists.append(labels)
    
    logger.info(f"Loaded {len(prompts)} prompts with {sum(len(r) for r in response_lists)} total responses")
    return prompts, response_lists, label_lists

def evaluate_alpha_values(
    prompts: List[str],
    response_lists: List[List[str]],
    label_lists: List[List[int]],
    alpha_values: List[float],
    device: str,
    limit_examples: int = None
) -> pd.DataFrame:
    """
    Evaluate multiple alpha values on the dataset
    
    Args:
        prompts: List of prompts
        response_lists: List of response lists (one per prompt)
        label_lists: List of label lists (one per prompt)
        alpha_values: List of alpha values to test
        device: Device to use for models
        limit_examples: Maximum number of examples to evaluate
        
    Returns:
        DataFrame with results for each alpha value
    """
    results = []
    
    # Limit examples if specified
    if limit_examples and limit_examples < len(prompts):
        prompts = prompts[:limit_examples]
        response_lists = response_lists[:limit_examples]
        label_lists = label_lists[:limit_examples]
    
    # Test each alpha value
    for alpha in alpha_values:
        logger.info(f"Testing alpha = {alpha}")
        
        # Initialize model with current alpha
        model = HarmonicBlendReward(
            device=device,
            alpha=alpha
        )
        
        # Process each prompt
        prompt_accuracies = []
        
        for prompt, responses, labels in zip(tqdm(prompts), response_lists, label_lists):
            # Score all responses for this prompt
            prompt_scores = {
                "similarity": [],
                "reward": [],
                "harmonic_blend": []
            }
            
            for response in responses:
                score_dict = model.score(prompt, response)
                
                # Save individual scores
                for key in prompt_scores:
                    prompt_scores[key].append(score_dict[key])
            
            # Compute accuracy for each score type
            accuracies = {}
            for key in prompt_scores:
                acc = compute_pairwise_accuracy(prompt_scores[key], labels)
                accuracies[key] = acc
            
            prompt_accuracies.append(accuracies)
            
        # Calculate overall accuracy for each score type
        avg_accuracies = {
            "alpha": alpha
        }
        
        for key in ["similarity", "reward", "harmonic_blend"]:
            values = [acc[key] for acc in prompt_accuracies]
            avg_accuracies[f"{key}_mean"] = np.mean(values)
            avg_accuracies[f"{key}_median"] = np.median(values)
            
        results.append(avg_accuracies)
        
        # Free memory
        free_memory()
        
    return pd.DataFrame(results)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Find optimal alpha for HarmonicBlendReward")
    parser.add_argument("--split", type=str, default="filtered", choices=["filtered", "raw"],
                       help="Dataset split to use")
    parser.add_argument("--min_alpha", type=float, default=0.0,
                       help="Minimum alpha value to test")
    parser.add_argument("--max_alpha", type=float, default=1.0,
                       help="Maximum alpha value to test")
    parser.add_argument("--steps", type=int, default=11,
                       help="Number of alpha values to test")
    parser.add_argument("--limit", type=int, default=100,
                       help="Maximum number of examples to process")
    parser.add_argument("--output_dir", type=str, default="./results/alpha_search",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_optimal_device()
    
    # Generate alpha values
    alpha_values = np.linspace(args.min_alpha, args.max_alpha, args.steps)
    logger.info(f"Testing {len(alpha_values)} alpha values: {alpha_values}")
    
    # Load RewardBench dataset
    prompts, response_lists, label_lists = load_rewardbench(
        split=args.split,
        limit=args.limit
    )
    
    # Evaluate all alpha values
    results_df = evaluate_alpha_values(
        prompts=prompts,
        response_lists=response_lists,
        label_lists=label_lists,
        alpha_values=alpha_values,
        device=device,
        limit_examples=args.limit
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, f"alpha_results_{args.split}.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Find optimal alpha
    optimal_alpha = results_df.loc[results_df["harmonic_blend_mean"].idxmax()]["alpha"]
    max_accuracy = results_df.loc[results_df["harmonic_blend_mean"].idxmax()]["harmonic_blend_mean"]
    
    logger.info(f"Optimal alpha = {optimal_alpha:.4f} (accuracy: {max_accuracy:.4f})")
    
    # Create visualizations
    try:
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Plot mean accuracy for each score type
        plt.plot(results_df["alpha"], results_df["similarity_mean"], marker='o', label="Similarity")
        plt.plot(results_df["alpha"], results_df["reward_mean"], marker='s', label="Reward")
        plt.plot(results_df["alpha"], results_df["harmonic_blend_mean"], marker='d', label="Harmonic Blend")
        
        # Add vertical line for optimal alpha
        plt.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.7)
        plt.text(optimal_alpha + 0.02, 0.5, f'Optimal α = {optimal_alpha:.3f}', rotation=90)
        
        # Format plot
        plt.xlabel("Alpha")
        plt.ylabel("Mean Pairwise Accuracy")
        plt.title(f"Accuracy vs Alpha ({args.split} split)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(args.output_dir, f"alpha_accuracy_{args.split}.png")
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
        
        # Create a simplified results summary
        summary = {
            "optimal_alpha": float(optimal_alpha),
            "max_accuracy": float(max_accuracy),
            "similarity_accuracy": float(results_df.loc[results_df["alpha"] == 0.0]["similarity_mean"].values[0]),
            "reward_accuracy": float(results_df.loc[results_df["alpha"] == 1.0]["reward_mean"].values[0]),
            "improvement_over_similarity": float(max_accuracy - results_df.loc[results_df["alpha"] == 0.0]["similarity_mean"].values[0]),
            "improvement_over_reward": float(max_accuracy - results_df.loc[results_df["alpha"] == 1.0]["reward_mean"].values[0]),
            "split": args.split,
            "examples_processed": len(prompts)
        }
        
        # Save summary
        summary_path = os.path.join(args.output_dir, f"summary_{args.split}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
    except Exception as e:
        logger.warning(f"Couldn't generate visualizations: {e}")
    
    logger.info("\nOptimal alpha search complete!")


if __name__ == "__main__":
    main()