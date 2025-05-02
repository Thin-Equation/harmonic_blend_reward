#!/usr/bin/env python
"""
Evaluate the HarmonicBlendReward model on the RewardBench dataset.

RewardBench is a benchmark for evaluating reward models on their ability to
rank responses based on quality. This script will load RewardBench, run the
harmonic blend model on the dataset, and compute various metrics.
"""

import os
import logging
import argparse
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple
from datasets import load_dataset

from models.core import HarmonicBlendReward
from utils.device_utils import get_optimal_device, free_memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_rewardbench(split: str = "filtered", limit: int = None) -> Tuple[List[str], List[List[str]], List[List[int]]]:
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

def compute_pairwise_accuracy(
    scores: List[float], 
    labels: List[int]
) -> float:
    """
    Compute pairwise accuracy for a set of responses
    
    Args:
        scores: Model scores for responses
        labels: Ground truth labels (higher is better)
        
    Returns:
        Pairwise accuracy
    """
    pairs_total = 0
    pairs_correct = 0
    
    # Compare all possible pairs
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            # Skip if labels are the same
            if labels[i] == labels[j]:
                continue
                
            pairs_total += 1
            
            # Check if model rankings match ground truth
            if (scores[i] > scores[j] and labels[i] > labels[j]) or \
               (scores[i] < scores[j] and labels[i] < labels[j]):
                pairs_correct += 1
    
    # Return accuracy
    if pairs_total == 0:
        return 1.0  # Default to perfect if no valid pairs
    
    return pairs_correct / pairs_total

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Evaluate HarmonicBlendReward on RewardBench")
    parser.add_argument("--split", type=str, default="filtered", choices=["filtered", "raw"],
                       help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Maximum number of examples to process")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight parameter for harmonic blend (0-1)")
    parser.add_argument("--output_dir", type=str, default="./results/rewardbench",
                       help="Directory to save results")
    parser.add_argument("--save_scores", action="store_true",
                       help="Save raw scores to a CSV file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_optimal_device()
    
    # Load the HarmonicBlendReward model
    logger.info(f"Initializing Harmonic Blend Reward model (alpha={args.alpha})")
    model = HarmonicBlendReward(
        device=device,
        alpha=args.alpha
    )
    
    # Load RewardBench dataset
    prompts, response_lists, label_lists = load_rewardbench(
        split=args.split,
        limit=args.limit
    )
    
    # Prepare for results
    all_results = []
    all_accuracies = []
    
    # Process each prompt
    for i, (prompt, responses, labels) in enumerate(zip(tqdm(prompts), response_lists, label_lists)):
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
            
            # Save all results
            all_results.append({
                "prompt": prompt,
                "response": response,
                **score_dict
            })
        
        # Compute accuracy for each score type
        accuracies = {}
        for key in prompt_scores:
            acc = compute_pairwise_accuracy(prompt_scores[key], labels)
            accuracies[f"{key}_accuracy"] = acc
        
        # Save accuracies
        accuracies["prompt_index"] = i
        all_accuracies.append(accuracies)
        
        # Free memory
        if i % 100 == 0:
            free_memory()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    accuracies_df = pd.DataFrame(all_accuracies)
    
    # Compute overall metrics
    metrics = {
        "score_type": [],
        "mean_accuracy": [],
        "median_accuracy": []
    }
    
    for key in ["similarity", "reward", "harmonic_blend"]:
        acc_key = f"{key}_accuracy"
        metrics["score_type"].append(key)
        metrics["mean_accuracy"].append(accuracies_df[acc_key].mean())
        metrics["median_accuracy"].append(accuracies_df[acc_key].median())
    
    metrics_df = pd.DataFrame(metrics)
    
    # Print results
    logger.info("\n=== RESULTS ===")
    logger.info(f"Dataset split: {args.split}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Total prompts processed: {len(prompts)}")
    logger.info("\nAccuracy by score type:")
    
    for _, row in metrics_df.iterrows():
        logger.info(f"  {row['score_type']}: {row['mean_accuracy']:.4f} (mean), {row['median_accuracy']:.4f} (median)")
    
    # Save results
    metrics_path = os.path.join(args.output_dir, f"metrics_{args.split}_alpha{args.alpha:.2f}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nMetrics saved to {metrics_path}")
    
    # Save accuracies
    accuracies_path = os.path.join(args.output_dir, f"accuracies_{args.split}_alpha{args.alpha:.2f}.csv")
    accuracies_df.to_csv(accuracies_path, index=False)
    logger.info(f"Accuracies saved to {accuracies_path}")
    
    # Optionally save all scores
    if args.save_scores:
        scores_path = os.path.join(args.output_dir, f"scores_{args.split}_alpha{args.alpha:.2f}.csv")
        results_df.to_csv(scores_path, index=False)
        logger.info(f"Raw scores saved to {scores_path}")
    
    # Create visualizations
    try:
        # Plot accuracy comparison
        plt_path = os.path.join(args.output_dir, f"accuracy_comparison_{args.split}_alpha{args.alpha:.2f}.png")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="score_type", y="mean_accuracy", data=metrics_df)
        plt.title(f"RewardBench Accuracy Comparison ({args.split} split)")
        plt.xlabel("Score Type")
        plt.ylabel("Mean Pairwise Accuracy")
        plt.ylim(0, 1)
        plt.savefig(plt_path)
        logger.info(f"Visualization saved to {plt_path}")
        
    except ImportError as e:
        logger.warning(f"Couldn't generate visualizations: {e}")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()