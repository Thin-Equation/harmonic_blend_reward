"""
Evaluation utilities for the harmonic blend reward model.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics import precision_recall_fscore_support

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RewardEvaluator:
    """
    Evaluator class for analyzing and benchmarking reward model outputs
    """
    
    def __init__(self, results: List[Dict[str, float]], threshold: float = 0.5):
        """
        Initialize the evaluator with reward model results
        
        Args:
            results: List of score dictionaries from the reward model
            threshold: Threshold for binary classification (default: 0.5)
        """
        self.results = results
        self.threshold = threshold
        self.df = pd.DataFrame(results)
        logger.info(f"Initialized evaluator with {len(results)} results")
        
    def summary_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for each score type
        
        Returns:
            Dictionary of statistics for each score type
        """
        stats = {}
        
        for column in self.df.columns:
            col_stats = {
                "mean": float(self.df[column].mean()),
                "median": float(self.df[column].median()),
                "std": float(self.df[column].std()),
                "min": float(self.df[column].min()),
                "max": float(self.df[column].max()),
                "25%": float(self.df[column].quantile(0.25)),
                "75%": float(self.df[column].quantile(0.75))
            }
            stats[column] = col_stats
            
        logger.info(f"Calculated summary statistics for {len(stats)} score types")
        return stats
        
    def compare_models(self, ground_truth: List[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare different scoring methods against ground truth
        
        Args:
            ground_truth: Optional list of binary ground truth labels
            
        Returns:
            Dictionary of metrics for each score type
        """
        metrics = {}
        
        # If ground truth is provided, calculate classification metrics
        if ground_truth is not None:
            if len(ground_truth) != len(self.results):
                raise ValueError(f"Ground truth length ({len(ground_truth)}) must match results length ({len(self.results)})")
                
            for column in self.df.columns:
                # Convert scores to binary predictions
                predictions = (self.df[column] > self.threshold).astype(int)
                
                # Calculate metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    ground_truth, predictions, average='binary'
                )
                
                metrics[column] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float((predictions == ground_truth).mean())
                }
        
        # Calculate correlations between different score types
        correlation_matrix = self.df.corr()
        
        # Add correlations to metrics
        for column in self.df.columns:
            if column not in metrics:
                metrics[column] = {}
                
            for other_column in self.df.columns:
                if column != other_column:
                    metrics[column][f"correlation_with_{other_column}"] = float(correlation_matrix.loc[column, other_column])
        
        logger.info(f"Calculated comparison metrics for {len(metrics)} score types")
        return metrics
        
    def score_distribution(self, bins: int = 20) -> Dict[str, Dict[str, List[float]]]:
        """
        Calculate histograms of score distributions
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            Dictionary of histogram data for each score type
        """
        distributions = {}
        
        for column in self.df.columns:
            hist, bin_edges = np.histogram(self.df[column], bins=bins)
            
            # Convert to lists for JSON serialization
            distributions[column] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
        
        logger.info(f"Calculated score distributions with {bins} bins")
        return distributions


def evaluate_reward_model(
    model,
    data_source: Union[str, List[Dict[str, Any]]],
    prompt_key: str = "prompt",
    response_key: str = "response",
    ground_truth_key: Optional[str] = None,
    limit: Optional[int] = None,
    threshold: float = 0.5
) -> Tuple[RewardEvaluator, pd.DataFrame]:
    """
    Convenience function to evaluate a reward model on a dataset
    
    Args:
        model: The reward model to evaluate
        data_source: Path to data file or list of examples
        prompt_key: Key for prompt field in the data
        response_key: Key for response field in the data
        ground_truth_key: Optional key for ground truth labels
        limit: Maximum number of examples to evaluate
        threshold: Classification threshold
        
    Returns:
        Tuple of (RewardEvaluator, DataFrame with results)
    """
    from data import load_prompt_response_pairs
    
    # Load data
    pairs = load_prompt_response_pairs(
        data_source, 
        prompt_key=prompt_key, 
        response_key=response_key, 
        limit=limit
    )
    
    # Get ground truth if specified
    ground_truth = None
    if ground_truth_key and isinstance(data_source, list):
        ground_truth = [example.get(ground_truth_key, 0) for example in data_source[:len(pairs)]]
    
    # Get model scores
    logger.info(f"Calculating scores for {len(pairs)} examples")
    scores = model.batch_score(pairs)
    
    # Create evaluator
    evaluator = RewardEvaluator(scores, threshold=threshold)
    
    # Create results DataFrame
    df = pd.DataFrame({
        "prompt": [p for p, _ in pairs],
        "response": [r for _, r in pairs],
        **{k: [s[k] for s in scores] for k in scores[0].keys()}
    })
    
    if ground_truth:
        df["ground_truth"] = ground_truth
        
    return evaluator, df