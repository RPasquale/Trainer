'''import logging
import argparse
from datasets import Dataset
from data_loader import load_dataset_by_config, get_data_loader
from tokenizer import tokenizer, data_collator
from config import datasets_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_datasets(datasets_config, dataset_name=None):
    for config in datasets_config:
        if dataset_name and config['name'] != dataset_name:
            continue
        
        try:
            logger.info(f"Testing dataset: {config['name']}")
            dataset, columns, task = load_dataset_by_config(config)
            
            # Check if columns exist in the dataset
            for column in columns:
                if column not in dataset.column_names:
                    raise ValueError(f"Column '{column}' not found in dataset '{config['name']}'")
            
            processed_dataset = get_data_loader(dataset, columns, task)
            
            # Tokenize a small subset of the dataset to check if tokenization works
            sample = processed_dataset.select(range(10))
            tokenized_sample = sample.map(lambda examples: tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=512), batched=True)
            
            # Add labels to tokenized sample
            def add_labels(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            
            tokenized_sample = tokenized_sample.map(add_labels, batched=True)
            
            # Ensure tokenized data has the same length
            def ensure_same_length(examples):
                min_length = min(len(examples['input_ids']), len(examples['labels']))
                examples['input_ids'] = examples['input_ids'][:min_length]
                examples['attention_mask'] = examples['attention_mask'][:min_length]
                examples['labels'] = examples['labels'][:min_length]
                return examples
            
            tokenized_sample = tokenized_sample.map(ensure_same_length, batched=True)
            
            # Check if tokenized sample has the expected columns
            if 'input_ids' not in tokenized_sample.column_names or 'labels' not in tokenized_sample.column_names:
                raise ValueError(f"Tokenization failed for dataset '{config['name']}'")
            
            logger.info(f"Dataset '{config['name']}' passed.")
        
        except Exception as e:
            logger.error(f"Error processing dataset '{config['name']}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test datasets for tokenization and loading")
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to test. If not specified, test all datasets.')
    args = parser.parse_args()
    
    test_datasets(datasets_config, dataset_name=args.dataset_name)
'''

import logging
import argparse
from datasets import Dataset
from data_loader import load_dataset_by_config, get_data_loader
from tokenizer import tokenizer, data_collator
from config import datasets_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_datasets(datasets_config, dataset_name=None):
    for config in datasets_config:
        if dataset_name and config['name'] != dataset_name:
            continue
        
        try:
            logger.info(f"Testing dataset: {config['name']}")
            dataset, columns, task = load_dataset_by_config(config)
            
            # Check if columns exist in the dataset
            for column in columns:
                if column not in dataset.column_names:
                    raise ValueError(f"Column '{column}' not found in dataset '{config['name']}'")
            
            processed_dataset = get_data_loader(dataset, columns, task)
            
            # Tokenize a small subset of the dataset to check if tokenization works
            sample = processed_dataset.select(range(10))
            tokenized_sample = sample.map(lambda examples: tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=512), batched=True)
            
            # Add labels to tokenized sample
            def add_labels(examples):
                examples["labels"] = examples["input_ids"].copy()
                return examples
            
            tokenized_sample = tokenized_sample.map(add_labels, batched=True)
            
            # Ensure tokenized data has the same length
            def ensure_same_length(examples):
                min_length = min(len(examples['input_ids']), len(examples['labels']))
                examples['input_ids'] = examples['input_ids'][:min_length]
                examples['attention_mask'] = examples['attention_mask'][:min_length]
                examples['labels'] = examples['labels'][:min_length]
                return examples
            
            tokenized_sample = tokenized_sample.map(ensure_same_length, batched=True)
            
            # Check if tokenized sample has the expected columns
            if 'input_ids' not in tokenized_sample.column_names or 'labels' not in tokenized_sample.column_names:
                raise ValueError(f"Tokenization failed for dataset '{config['name']}'")
            
            logger.info(f"Dataset '{config['name']}' passed.")
        
        except Exception as e:
            logger.error(f"Error processing dataset '{config['name']}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test datasets for tokenization and loading")
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to test. If not specified, test all datasets.')
    args = parser.parse_args()
    
    test_datasets(datasets_config, dataset_name=args.dataset_name)

