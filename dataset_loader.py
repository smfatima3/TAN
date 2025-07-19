"""
Dataset Loading and Preprocessing for Topoformer Experiments
Supports multiple domains: Bug Localization, Scientific Papers, Legal Documents
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopoformerDataset(Dataset):
    """Base dataset class for Topoformer experiments"""
    
    def __init__(self,
                 data: List[Dict],
                 tokenizer: AutoTokenizer,
                 max_length: int = 256,
                 task_type: str = 'classification'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        # Add labels based on task type
        if self.task_type == 'classification':
            result['labels'] = torch.tensor(item['label'], dtype=torch.long)
        elif self.task_type == 'multi_label':
            result['labels'] = torch.tensor(item['labels'], dtype=torch.float)
        
        return result


class DatasetLoader:
    """Handles loading and preprocessing of different datasets"""
    
    def __init__(self, cache_dir: Optional[str] = './cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_bug_localization_dataset(self,
                                     subset_size: Optional[int] = None,
                                     test_size: float = 0.2) -> Tuple[Dataset, Dataset, Dict]:
        """Load JetBrains bug localization dataset"""
        logger.info("Loading JetBrains bug localization dataset...")

        # Load the base dataset. The splits will be accessed directly.
        dataset = load_dataset('JetBrains-Research/lca-bug-localization', 'py',
            cache_dir=self.cache_dir
        )

        # Prepare data
        data = []
        label_map = {}
        label_counter = 0

        # Iterate over the correct splits: 'dev', 'train', 'test'
        for split in ['dev', 'train', 'test']:
            if split not in dataset:
                continue

            logger.info(f"Checking split: '{split}'")
            for item in tqdm(dataset[split], desc=f"Processing bug reports from {split}"):
                # Use the correct field name 'changed_files'
                if not item.get('changed_files'):
                    continue
                
                # Use the correct fields for text: 'issue_title' and 'issue_body'
                text = f"{item.get('issue_title', '')} {item.get('issue_body', '')}"
                
                # Iterate over the 'changed_files' list
                for file_path in item['changed_files']:
                    if file_path not in label_map:
                        label_map[file_path] = label_counter
                        label_counter += 1
                    
                    data.append({
                        'text': text,
                        'label': label_map[file_path],
                        'file_path': file_path,
                        'bug_id': item.get('issue_url', '') # Use a more unique ID
                    })

        if not data:
            raise ValueError(
                "Processed data is empty even after using the correct dataset structure. "
                "Please check the dataset content on Hugging Face."
            )

        # Limit dataset size if specified
        if subset_size and len(data) > subset_size:
            indices = np.random.choice(range(len(data)), subset_size, replace=False)
            data = [data[i] for i in indices]
            
        # We can't stratify if some classes have only one member.
        # We will try to stratify, but fall back to a random split if it fails.
        try:
            train_data, val_data = train_test_split(
                data,
                test_size=test_size,
                random_state=42,
                stratify=[d['label'] for d in data]
            )
        except ValueError:
            logger.warning(
                "Stratified split failed, likely due to single-member classes. "
                "Falling back to a random split."
            )
            train_data, val_data = train_test_split(
                data,
                test_size=test_size,
                random_state=42
            )
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Create datasets
        train_dataset = TopoformerDataset(train_data, tokenizer, task_type='classification')
        val_dataset = TopoformerDataset(val_data, tokenizer, task_type='classification')
        
        metadata = {
            'num_labels': len(label_map),
            'label_map': label_map,
            'task_type': 'bug_localization',
            'train_size': len(train_data),
            'val_size': len(val_data)
        }
        
        logger.info(f"Loaded {len(train_data)} train and {len(val_data)} val samples")
        logger.info(f"Number of unique file paths (labels): {len(label_map)}")
        
        return train_dataset, val_dataset, metadata
    
    def load_multi_eurlex_dataset(self,
                                 language_pair: str = 'en',
                                 subset_size: Optional[int] = None,
                                 test_size: float = 0.2) -> Tuple[Dataset, Dataset, Dict]:
        """Load Multi-EURLEX legal document dataset"""
        logger.info(f"Loading Multi-EURLEX dataset for {language_pair}...")
        
        # Load dataset
        dataset = load_dataset(
            "multi_eurlex",
            language_pair,
            cache_dir=self.cache_dir
        )
        
        # Prepare data with hierarchical labels
        data = []
        all_labels = set()
        
        for item in tqdm(dataset['train'], desc="Processing legal documents"):
            text = item['text']
            labels = item['labels']  # Multi-label classification
            
            # Convert to binary vector
            all_labels.update(labels)
            
            data.append({
                'text': text,
                'labels': labels,
                'celex_id': item.get('celex_id', '')
            })
        
        # Create label mapping
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        # Convert labels to binary vectors
        for item in data:
            label_vector = [0] * len(label_to_idx)
            for label in item['labels']:
                label_vector[label_to_idx[label]] = 1
            item['labels'] = label_vector
        
        # Limit dataset size
        if subset_size and len(data) > subset_size:
            data = data[:subset_size]
        
        # Split data
        train_data, val_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # Create datasets
        train_dataset = TopoformerDataset(train_data, tokenizer, task_type='multi_label')
        val_dataset = TopoformerDataset(val_data, tokenizer, task_type='multi_label')
        
        metadata = {
            'num_labels': len(label_to_idx),
            'label_map': label_to_idx,
            'task_type': 'multi_label_classification',
            'train_size': len(train_data),
            'val_size': len(val_data)
        }
        
        logger.info(f"Loaded {len(train_data)} train and {len(val_data)} val samples")
        logger.info(f"Number of unique labels: {len(label_to_idx)}")
        
        return train_dataset, val_dataset, metadata
    
    def load_arxiv_papers_dataset(self,
                                 subset_size: Optional[int] = None,
                                 test_size: float = 0.2) -> Tuple[Dataset, Dataset, Dict]:
        """Load ArXiv papers dataset for hierarchical document understanding"""
        logger.info("Loading ArXiv papers dataset...")
        
        # For ArXiv, we'll create a classification task based on paper categories
        # In practice, you would load actual ArXiv data
        
        categories = [
            'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE',
            'math.CO', 'math.PR', 'math.ST', 
            'physics.comp-ph', 'physics.data-an'
        ]
        
        # Generate synthetic ArXiv-like data for demonstration
        data = []
        np.random.seed(42)
        
        for i in range(subset_size or 10000):
            # Simulate hierarchical structure in papers
            category = np.random.choice(categories)
            
            # Create text with hierarchical structure
            title = f"Paper {i}: Advanced Methods in {category}"
            abstract = f"We present novel approaches for {category} problems. "
            abstract += "Our method shows significant improvements over baselines. "
            abstract += f"Experiments on {category} benchmarks validate our approach."
            
            text = f"{title}\n\nAbstract: {abstract}"
            
            data.append({
                'text': text,
                'label': categories.index(category),
                'category': category,
                'paper_id': f'arxiv_{i}'
            })
        
        # Split data
        train_data, val_data = train_test_split(
            data, test_size=test_size, random_state=42, 
            stratify=[d['label'] for d in data]
        )
        
        # Create tokenizer (using BigBird tokenizer since it's trained on ArXiv)
        tokenizer = AutoTokenizer.from_pretrained('google/bigbird-pegasus-large-arxiv')
        
        # Create datasets
        train_dataset = TopoformerDataset(train_data, tokenizer, max_length=512)
        val_dataset = TopoformerDataset(val_data, tokenizer, max_length=512)
        
        metadata = {
            'num_labels': len(categories),
            'categories': categories,
            'task_type': 'arxiv_classification',
            'train_size': len(train_data),
            'val_size': len(val_data)
        }
        
        logger.info(f"Loaded {len(train_data)} train and {len(val_data)} val samples")
        logger.info(f"Number of categories: {len(categories)}")
        
        return train_dataset, val_dataset, metadata
    
    def load_wikipedia_dataset(self,
                              subset_size: Optional[int] = None,
                              test_size: float = 0.2) -> Tuple[Dataset, Dataset, Dict]:
        """Load Wikipedia dataset for hierarchical document understanding"""
        logger.info("Loading Wikipedia dataset...")
        
        # Load dataset
        dataset = load_dataset(
            "howey/wiki_en",
            split="train",
            cache_dir=self.cache_dir
        )
        
        # Process Wikipedia articles with section hierarchy
        data = []
        category_map = {}
        category_counter = 0
        
        for idx, item in enumerate(tqdm(
            dataset.select(range(min(subset_size or 10000, len(dataset)))),
            desc="Processing Wikipedia articles"
        )):
            text = item.get('text', '')
            title = item.get('title', '')
            
            # Extract category from article (simplified)
            if 'Science' in title or 'science' in text[:200]:
                category = 'Science'
            elif 'History' in title or 'history' in text[:200]:
                category = 'History'
            elif 'Technology' in title or 'technology' in text[:200]:
                category = 'Technology'
            elif 'Art' in title or 'art' in text[:200]:
                category = 'Art'
            else:
                category = 'General'
            
            if category not in category_map:
                category_map[category] = category_counter
                category_counter += 1
            
            # Take first 512 characters for efficiency
            text_sample = f"{title}\n\n{text[:512]}"
            
            data.append({
                'text': text_sample,
                'label': category_map[category],
                'category': category,
                'title': title
            })
        
        # Split data
        train_data, val_data = train_test_split(
            data, test_size=test_size, random_state=42,
            stratify=[d['label'] for d in data]
        )
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create datasets
        train_dataset = TopoformerDataset(train_data, tokenizer)
        val_dataset = TopoformerDataset(val_data, tokenizer)
        
        metadata = {
            'num_labels': len(category_map),
            'category_map': category_map,
            'task_type': 'wikipedia_classification',
            'train_size': len(train_data),
            'val_size': len(val_data)
        }
        
        logger.info(f"Loaded {len(train_data)} train and {len(val_data)} val samples")
        logger.info(f"Number of categories: {len(category_map)}")
        
        return train_dataset, val_dataset, metadata


def create_data_loaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       batch_size: int = 16,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def test_dataset_loading():
    """Test dataset loading functionality"""
    print("Testing Dataset Loading")
    print("=" * 50)
    
    loader = DatasetLoader()
    
    # Test with small subset
    print("\n1. Testing Bug Localization Dataset (100 samples)")
    try:
        train_dataset, val_dataset, metadata = loader.load_bug_localization_dataset(
            subset_size=100
        )
        print(f"✓ Bug localization: {metadata['train_size']} train, {metadata['val_size']} val")
        
        # Test data loader
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, batch_size=4
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"  Batch shapes: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
    except Exception as e:
        print(f"✗ Bug localization failed: {e}")
    
    print("\n2. Testing Multi-EURLEX Dataset (100 samples)")
    try:
        train_dataset, val_dataset, metadata = loader.load_multi_eurlex_dataset(
            subset_size=100
        )
        print(f"✓ Multi-EURLEX: {metadata['train_size']} train, {metadata['val_size']} val")
    except Exception as e:
        print(f"✗ Multi-EURLEX failed: {e}")
    
    print("\n3. Testing ArXiv Dataset (100 samples)")
    try:
        train_dataset, val_dataset, metadata = loader.load_arxiv_papers_dataset(
            subset_size=100
        )
        print(f"✓ ArXiv: {metadata['train_size']} train, {metadata['val_size']} val")
    except Exception as e:
        print(f"✗ ArXiv failed: {e}")
    
    print("\n4. Testing Wikipedia Dataset (100 samples)")
    try:
        train_dataset, val_dataset, metadata = loader.load_wikipedia_dataset(
            subset_size=100
        )
        print(f"✓ Wikipedia: {metadata['train_size']} train, {metadata['val_size']} val")
    except Exception as e:
        print(f"✗ Wikipedia failed: {e}")
    
    print("\nDataset loading tests completed!")


if __name__ == "__main__":
    test_dataset_loading()
