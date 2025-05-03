#!/usr/bin/env python
"""
python find_optimal_alpha.pypython find_optimal_alpha.pyFind the optimal alpha parameter for HarmonicBlendReward using a custom dataset.

This script evaluates the HarmonicBlendReward model with different alpha values
to find the optimal balance between similarity and reward scores using the
validation split of the custom dataset.
"""

import os
import logging
import argparse
import json
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from models.core import HarmonicBlendReward
from utils.device_utils import get_optimal_device, free_memory
from datasets import load_dataset, Dataset

# Import functions from evaluate_rewardbench.py
from evaluate_rewardbench import compute_pairwise_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_custom_dataset(path: str, random_seed: int = 42) -> Dict[str, Dataset]:
    """
    Load the custom combined dataset and split it into train, validation, and test
    
    Args:
        path: Path to the combined dataset JSONL file
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, validation and test datasets
    """
    logger.info(f"Loading custom dataset from {path}")
    
    # Load dataset
    full_dataset = load_dataset("json", data_files=path)["train"]
    logger.info(f"Loaded {len(full_dataset)} examples from {path}")
    
    # Log the distribution of sources
    source_counts = {}
    for source in full_dataset["source"]:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    logger.info("Dataset source distribution:")
    for source, count in source_counts.items():
        logger.info(f"  - {source}: {count} examples")
    
    # Properly split the dataset using Hugging Face's methods
    # Shuffle the dataset first
    full_dataset = full_dataset.shuffle(seed=random_seed)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    validation_test_size = int(0.3 * total_size)
    validation_size = int(0.5 * validation_test_size)
    test_size = validation_test_size - validation_size
    
    # Create the splits
    train_dataset = full_dataset.select(range(total_size - validation_test_size))
    temp_remaining = full_dataset.select(range(total_size - validation_test_size, total_size))
    validation_dataset = temp_remaining.select(range(validation_size))
    test_dataset = temp_remaining.select(range(validation_size, validation_size + test_size))
    
    logger.info(f"Split dataset into train ({len(train_dataset)}), validation ({len(validation_dataset)}), and test ({len(test_dataset)}) sets")
    
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    }

def prepare_pairwise_comparisons(
    dataset: Dataset, 
    num_comparisons: int = 400
) -> Tuple[List[str], List[List[str]], List[List[int]]]:
    """
    Create pairwise comparisons from the dataset by generating pairs of responses
    for each prompt. Since these are not ranked by humans, we use the source
    to create artificial preferences: truthfulqa and scholarbench responses preferred
    over ms_marco and xnli responses.
    
    Args:
        dataset: Dataset with prompts and responses
        num_comparisons: Number of comparisons to generate
        
    Returns:
        Tuple of (prompts, response_lists, label_lists)
    """
    data = [(item["prompt"], item["response"], item["source"]) for item in dataset]
    
    # Define preference order for sources
    source_preference = {
        "truthfulqa": 3,  # Highest preference
        "scholarbench": 2,
        "ms_marco": 1,
        "xnli": 0        # Lowest preference
    }
    
    prompts = []
    response_lists = []
    label_lists = []
    
    # Group data by prompt
    prompt_groups = {}
    for prompt, response, source in data:
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append((response, source))
    
    # Generate comparison pairs for each prompt
    comparisons_created = 0
    
    # First, try to create comparisons for prompts with multiple responses
    for prompt, responses in prompt_groups.items():
        if len(responses) >= 2 and comparisons_created < num_comparisons:
            # Create a comparison pair
            selected_responses = random.sample(responses, 2)
            
            # Assign preferences based on source
            response1, source1 = selected_responses[0]
            response2, source2 = selected_responses[1]
            
            pref1 = source_preference.get(source1, 0)
            pref2 = source_preference.get(source2, 0)
            
            if pref1 == pref2:
                # If same preference, randomly choose
                if random.random() < 0.5:
                    chosen, rejected = response1, response2
                    labels = [1, 0]  # 1 for chosen, 0 for rejected
                else:
                    chosen, rejected = response2, response1
                    labels = [1, 0]  # 1 for chosen, 0 for rejected
            elif pref1 > pref2:
                chosen, rejected = response1, response2
                labels = [1, 0]
            else:
                chosen, rejected = response2, response1
                labels = [1, 0]
                
            prompts.append(prompt)
            response_lists.append([chosen, rejected])
            label_lists.append(labels)
            comparisons_created += 1
    
    # If we couldn't create enough comparisons, create synthetic ones
    while comparisons_created < num_comparisons:
        # Pick two random entries
        entry1, entry2 = random.sample(data, 2)
        prompt1, response1, source1 = entry1
        _, response2, source2 = entry2
        
        # Use the first prompt
        prompt = prompt1
        
        pref1 = source_preference.get(source1, 0)
        pref2 = source_preference.get(source2, 0)
        
        if pref1 == pref2:
            # If same preference, randomly choose
            if random.random() < 0.5:
                chosen, rejected = response1, response2
                labels = [1, 0]
            else:
                chosen, rejected = response2, response1
                labels = [1, 0]
        elif pref1 > pref2:
            chosen, rejected = response1, response2
            labels = [1, 0]
        else:
            chosen, rejected = response2, response1
            labels = [1, 0]
            
        prompts.append(prompt)
        response_lists.append([chosen, rejected])
        label_lists.append(labels)
        comparisons_created += 1
    
    logger.info(f"Created {len(prompts)} pairwise comparisons for evaluation")
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
    parser.add_argument("--dataset", type=str, default="data/combined_dataset.jsonl",
                       help="Path to the custom dataset JSONL file")
    parser.add_argument("--min_alpha", type=float, default=0.0,
                       help="Minimum alpha value to test")
    parser.add_argument("--max_alpha", type=float, default=1.0,
                       help="Maximum alpha value to test")
    parser.add_argument("--steps", type=int, default=11,
                       help="Number of alpha values to test")
    parser.add_argument("--comparisons", type=int, default=400,
                       help="Number of pairwise comparisons to generate for evaluation")
    parser.add_argument("--output_dir", type=str, default="./results/alpha_search",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = get_optimal_device()
    
    # Generate alpha values
    alpha_values = np.linspace(args.min_alpha, args.max_alpha, args.steps)
    logger.info(f"Testing {len(alpha_values)} alpha values: {alpha_values}")
    
    # Load custom dataset and split into train/val/test
    datasets = load_custom_dataset(args.dataset, random_seed=args.seed)
    
    # Prepare pairwise comparisons from validation set
    prompts, response_lists, label_lists = prepare_pairwise_comparisons(
        datasets["validation"], 
        num_comparisons=args.comparisons
    )
    
    # Evaluate all alpha values
    results_df = evaluate_alpha_values(
        prompts=prompts,
        response_lists=response_lists,
        label_lists=label_lists,
        alpha_values=alpha_values,
        device=device
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "alpha_results_custom.csv")
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
        plt.title(f"Accuracy vs Alpha (custom dataset)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(args.output_dir, "alpha_accuracy_custom.png")
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
            "dataset": "custom",
            "examples_processed": len(prompts)
        }
        
        # Save summary
        summary_path = os.path.join(args.output_dir, "summary_custom.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
    except Exception as e:
        logger.warning(f"Couldn't generate visualizations: {e}")
    
    logger.info("\nOptimal alpha search complete!")


if __name__ == "__main__":
    main()