"""
Visualization utilities for reward model evaluation results.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_score_distributions(results_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10), 
                             bins: int = 30, save_path: Optional[str] = None):
    """
    Plot histograms of score distributions
    
    Args:
        results_df: DataFrame with score columns
        figsize: Figure size (width, height)
        bins: Number of histogram bins
        save_path: Optional path to save the figure
    """
    # Get score columns (typically float columns)
    score_columns = results_df.select_dtypes(include=['float64', 'float32']).columns
    
    # Create figure with subplots
    n_cols = min(3, len(score_columns))
    n_rows = (len(score_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot histograms
    for i, col in enumerate(score_columns):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(results_df[col], bins=bins, kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(len(score_columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved score distribution plot to {save_path}")
    
    plt.show()

def plot_correlation_heatmap(results_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None):
    """
    Plot a heatmap of score correlations
    
    Args:
        results_df: DataFrame with score columns
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    # Get numeric columns
    numeric_df = results_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        annot=True, 
        mask=mask, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Correlation between Different Scores', fontsize=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved correlation heatmap to {save_path}")
        
    plt.show()

def plot_scatter_matrix(results_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 12),
                       save_path: Optional[str] = None):
    """
    Plot a scatter matrix to show relationships between different scores
    
    Args:
        results_df: DataFrame with score columns
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    # Get score columns
    score_columns = results_df.select_dtypes(include=['float64', 'float32']).columns
    
    # Create pairplot
    g = sns.pairplot(results_df[score_columns], diag_kind='kde', height=2.5)
    g.fig.suptitle('Relationships Between Different Scores', y=1.02, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved scatter matrix to {save_path}")
        
    plt.show()

def plot_benchmark_comparison(metrics: Dict[str, Dict[str, float]], figsize: Tuple[int, int] = (12, 6),
                             metric_keys: List[str] = None, save_path: Optional[str] = None):
    """
    Plot a bar chart comparing metrics across different score types
    
    Args:
        metrics: Dictionary of metrics from RewardEvaluator.compare_models()
        figsize: Figure size (width, height)
        metric_keys: List of metric keys to plot (e.g., ['precision', 'recall', 'f1'])
        save_path: Optional path to save the figure
    """
    # If metric keys not specified, use common metrics
    if metric_keys is None:
        # Try to find common metrics across all score types
        all_keys = set()
        for score_type in metrics:
            all_keys.update(metrics[score_type].keys())
        
        # Filter to common classification metrics
        common_metrics = ['precision', 'recall', 'f1', 'accuracy']
        metric_keys = [key for key in common_metrics if key in all_keys]
        
        # If no common metrics found, use first 4 metrics from first score type
        if not metric_keys and metrics:
            first_score = list(metrics.keys())[0]
            metric_keys = list(metrics[first_score].keys())[:4]
    
    # Create DataFrame for plotting
    plot_data = []
    for score_type, metric_dict in metrics.items():
        for metric in metric_keys:
            if metric in metric_dict:
                plot_data.append({
                    'Score Type': score_type,
                    'Metric': metric,
                    'Value': metric_dict[metric]
                })
    
    if not plot_data:
        logger.warning("No metrics data available for plotting")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create bar chart
    plt.figure(figsize=figsize)
    sns.barplot(x='Score Type', y='Value', hue='Metric', data=plot_df)
    
    # Add labels and title
    plt.title('Comparison of Metrics Across Score Types', fontsize=15)
    plt.xlabel('Score Type')
    plt.ylabel('Value')
    
    # Format legend and layout
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved benchmark comparison plot to {save_path}")
        
    plt.show()

def create_report(results_df: pd.DataFrame, evaluator = None, 
                 output_dir: str = './reports', prefix: str = 'reward_model_report'):
    """
    Generate a comprehensive visual report from evaluation results
    
    Args:
        results_df: DataFrame with evaluation results
        evaluator: Optional RewardEvaluator instance
        output_dir: Directory to save report files
        prefix: Prefix for report filenames
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot score distributions
    dist_path = os.path.join(output_dir, f"{prefix}_distributions_{timestamp}.png")
    plot_score_distributions(results_df, save_path=dist_path)
    
    # Plot correlation heatmap
    corr_path = os.path.join(output_dir, f"{prefix}_correlations_{timestamp}.png")
    plot_correlation_heatmap(results_df, save_path=corr_path)
    
    # Plot scatter matrix
    scatter_path = os.path.join(output_dir, f"{prefix}_scatter_{timestamp}.png")
    plot_scatter_matrix(results_df, save_path=scatter_path)
    
    # If evaluator is provided, plot benchmark comparison
    if evaluator and hasattr(evaluator, 'compare_models'):
        metrics = evaluator.compare_models()
        bench_path = os.path.join(output_dir, f"{prefix}_benchmark_{timestamp}.png")
        plot_benchmark_comparison(metrics, save_path=bench_path)
        
    # Export results to CSV
    csv_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Exported results to {csv_path}")
    
    # If evaluator has summary stats, export those too
    if evaluator and hasattr(evaluator, 'summary_stats'):
        stats = evaluator.summary_stats()
        stats_df = pd.DataFrame({
            k1: {k2: v for k2, v in v1.items()} 
            for k1, v1 in stats.items()
        })
        stats_path = os.path.join(output_dir, f"{prefix}_stats_{timestamp}.csv")
        stats_df.to_csv(stats_path)
        logger.info(f"Exported summary statistics to {stats_path}")
        
    logger.info(f"Created report files in {output_dir}")
    return os.path.join(output_dir, f"{prefix}_*_{timestamp}.*")