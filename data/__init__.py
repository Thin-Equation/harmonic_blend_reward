"""
Data loading and processing utilities for the harmonic blend reward model.
"""

import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from datasets import Dataset, load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSONL data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_prompt_response_pairs(
    source: Union[str, List[Dict[str, Any]]],
    prompt_key: str = "prompt",
    response_key: str = "response",
    limit: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Load prompt-response pairs from various data sources
    
    Args:
        source: File path (JSONL, CSV) or HF dataset name or list of dictionaries
        prompt_key: Key for prompt field in the data
        response_key: Key for response field in the data
        limit: Maximum number of examples to load (None for all)
        
    Returns:
        List of (prompt, response) tuples
    """
    data = []
    
    # Handle different source types
    if isinstance(source, str):
        # Check if it's a file path
        if os.path.exists(source):
            # Determine file type
            if source.endswith('.jsonl'):
                raw_data = load_jsonl(source)
                logger.info(f"Loaded {len(raw_data)} examples from JSONL file: {source}")
            elif source.endswith('.csv'):
                df = pd.read_csv(source)
                raw_data = df.to_dict('records')
                logger.info(f"Loaded {len(raw_data)} examples from CSV file: {source}")
            elif source.endswith('.json'):
                with open(source, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                logger.info(f"Loaded {len(raw_data)} examples from JSON file: {source}")
            else:
                raise ValueError(f"Unsupported file format: {source}")
        else:
            # Try loading as a Hugging Face dataset
            try:
                dataset = load_dataset(source)
                if isinstance(dataset, dict):
                    # Get the first split (usually 'train')
                    split = list(dataset.keys())[0]
                    raw_data = dataset[split].to_list()
                else:
                    raw_data = dataset.to_list()
                logger.info(f"Loaded {len(raw_data)} examples from Hugging Face dataset: {source}")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise ValueError(f"Could not load data from: {source}")
    else:
        # Assume it's already a list of dictionaries
        raw_data = source
        logger.info(f"Using {len(raw_data)} provided examples")
    
    # Extract prompt-response pairs
    for item in raw_data:
        if prompt_key in item and response_key in item:
            data.append((item[prompt_key], item[response_key]))
            
            # Apply limit if specified
            if limit is not None and len(data) >= limit:
                break
                
    logger.info(f"Extracted {len(data)} valid prompt-response pairs")
    return data

def create_dataset_from_pairs(
    pairs: List[Tuple[str, str]],
    scores: List[Dict[str, float]] = None
) -> Dataset:
    """
    Create a Hugging Face dataset from prompt-response pairs and optional scores
    
    Args:
        pairs: List of (prompt, response) tuples
        scores: Optional list of score dictionaries
        
    Returns:
        Hugging Face Dataset
    """
    prompts, responses = zip(*pairs) if pairs else ([], [])
    
    data_dict = {
        "prompt": list(prompts),
        "response": list(responses),
    }
    
    # Add scores if provided
    if scores and len(scores) == len(pairs):
        for key in scores[0].keys():
            data_dict[key] = [s[key] for s in scores]
    
    return Dataset.from_dict(data_dict)