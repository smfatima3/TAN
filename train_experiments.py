"""
Main Training Script for Topoformer Experiments
Handles training, evaluation, and comparison with baselines
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import json
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import wandb
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer  # Added this import

# Import our modules
from topoformer_complete import (
    TopoformerConfig, 
    TopoformerForSequenceClassification,
    count_parameters
)
from dataset_loader import DatasetLoader, create_data_loaders
from baseline_models import create_baseline_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Model settings
    model_type: str = 'topoformer'  # topoformer, bert, codebert, longformer
    dataset_name: str = 'bug_localization'  # bug_localization, multi_eurlex, arxiv, wikipedia
    
    # Training settings
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data settings
    max_seq_length: int = 256
    train_samples: Optional[int] = 10000
    val_samples: Optional[int] = 2000
    
    # Hardware settings
    device: str = 'cuda'
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Experiment settings
    seed: int = 42
    save_dir: str = './experiments'
    use_wandb: bool = False
    experiment_name: str = 'topoformer_experiment'
    
    # Topoformer specific
    topoformer_layers: int = 6
    topoformer_heads: int = 12
    topoformer_embed_dim: int = 768
    k_neighbors: int = 32
    max_homology_dim: int = 2


class ExperimentTracker:
    """Track experiment metrics and results"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'epoch_times': [],
            'memory_usage': []
        }
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="topoformer-aaai",
                name=config.experiment_name,
                config=asdict(config)
            )
    
    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics for current step"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
    
    def save_results(self):
        """Save experiment results"""
        results = {
            'config': asdict(self.config),
            'metrics': self.metrics,
            'best_val_f1': max(self.metrics['val_f1']) if self.metrics['val_f1'] else 0,
            'final_val_f1': self.metrics['val_f1'][-1] if self.metrics['val_f1'] else 0,
            'total_training_time': sum(self.metrics['epoch_times']),
            'avg_memory_gb': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        }
        
        # Save to file
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.config.save_dir, 
            f"{self.config.experiment_name}_results.json"
        )
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
        return results


class Trainer:
    """Main trainer class for all models"""
    
    def __init__(self, model: nn.Module, config: ExperimentConfig, 
                 num_labels: int, tracker: ExperimentTracker):
        self.model = model
        self.config = config
        self.num_labels = num_labels
        self.tracker = tracker
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        epoch_start = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            
            attention_mask_int = batch['attention_mask'].to(self.device)
            attention_mask = (attention_mask_int == 0)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    if self.config.model_type == 'topoformer':
                        outputs = self.model(input_ids, attention_mask, labels)
                        loss = outputs['loss']
                        logits = outputs['logits']
                    else:
                        outputs = self.model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        logits = outputs.logits
            else:
                if self.config.model_type == 'topoformer':
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                else:
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.config.gradient_accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            memory_gb = 0
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_f1': f1,
            'epoch_time': epoch_time,
            'memory_gb': memory_gb
        }
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.model_type == 'topoformer':
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                else:
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training loop"""
        best_f1 = 0
        best_epoch = 0
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model: {self.config.model_type}")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            self.tracker.log_metrics(epoch_metrics, epoch)
            
            # Track memory usage
            self.tracker.metrics['epoch_times'].append(train_metrics['epoch_time'])
            self.tracker.metrics['memory_usage'].append(train_metrics['memory_gb'])
            
            # Print progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train F1: {train_metrics['train_f1']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val F1: {val_metrics['val_f1']:.4f}, "
                f"Time: {train_metrics['epoch_time']:.2f}s"
            )
            
            # Save best model
            if val_metrics['val_f1'] > best_f1:
                best_f1 = val_metrics['val_f1']
                best_epoch = epoch + 1
                
                # Save model
                save_path = os.path.join(
                    self.config.save_dir,
                    f"{self.config.experiment_name}_best_model.pt"
                )
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
        
        logger.info(f"Training completed. Best F1: {best_f1:.4f} at epoch {best_epoch}")
        
        return {
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'final_metrics': {**train_metrics, **val_metrics}
        }


def create_model(config: ExperimentConfig, num_labels: int, tokenizer=None) -> nn.Module:
    """Create model based on configuration"""
    
    if config.model_type == 'topoformer':
        # Get vocab size from tokenizer if available, otherwise use default
        vocab_size = len(tokenizer) if tokenizer else 50000
        
        topo_config = TopoformerConfig(
            vocab_size=vocab_size,
            embed_dim=config.topoformer_embed_dim,
            num_layers=config.topoformer_layers,
            num_heads=config.topoformer_heads,
            k_neighbors=config.k_neighbors,
            max_homology_dim=config.max_homology_dim,
            max_seq_len=max(512, config.max_seq_length),  # Ensure sufficient length
            mixed_precision=config.mixed_precision,
            gradient_checkpointing=True
        )
        model = TopoformerForSequenceClassification(topo_config, num_labels)
    else:
        baseline = create_baseline_model(config.model_type, num_labels, config.device)
        model = baseline.model
    
    return model


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment"""
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Add CUDA debugging if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable better CUDA error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    loader = DatasetLoader()
    
    if config.dataset_name == 'bug_localization':
        train_dataset, val_dataset, metadata = loader.load_bug_localization_dataset(
            subset_size=config.train_samples
        )
        # Get tokenizer for vocab size
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    elif config.dataset_name == 'multi_eurlex':
        train_dataset, val_dataset, metadata = loader.load_multi_eurlex_dataset(
            subset_size=config.train_samples
        )
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    elif config.dataset_name == 'arxiv':
        train_dataset, val_dataset, metadata = loader.load_arxiv_papers_dataset(
            subset_size=config.train_samples
        )
        tokenizer = AutoTokenizer.from_pretrained('google/bigbird-pegasus-large-arxiv')
    elif config.dataset_name == 'wikipedia':
        train_dataset, val_dataset, metadata = loader.load_wikipedia_dataset(
            subset_size=config.train_samples
        )
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    # Create data loaders with error handling
    try:
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset,
            batch_size=config.batch_size,
            num_workers=0 if config.device == 'cuda' else 4  # Avoid multiprocessing issues with CUDA
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        # Try with num_workers=0
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset,
            batch_size=config.batch_size,
            num_workers=0
        )
    
    # Create model with tokenizer
    logger.info(f"Creating model: {config.model_type}")
    model = create_model(config, metadata['num_labels'], tokenizer)
    
    # Log model info
    param_count = count_parameters(model) if hasattr(model, 'parameters') else 0
    logger.info(f"Model parameters: {param_count:,}")
    
    # Create trainer
    trainer = Trainer(model, config, metadata['num_labels'], tracker)
    
    # Train model
    train_results = trainer.train(train_loader, val_loader)
    
    # Save results
    experiment_results = tracker.save_results()
    experiment_results['train_results'] = train_results
    experiment_results['metadata'] = metadata
    
    return experiment_results


def run_hardware_benchmark(hardware_configs: List[Dict]) -> Dict:
    """Run experiments on different hardware configurations"""
    results = {}
    
    for hw_config in hardware_configs:
        logger.info(f"\nRunning on {hw_config['name']}...")
        
        # Adjust batch size and samples based on hardware
        config = ExperimentConfig(
            model_type='topoformer',
            dataset_name='bug_localization',
            batch_size=hw_config['batch_size'],
            train_samples=hw_config['train_samples'],
            num_epochs=hw_config['num_epochs'],
            device=hw_config['device'],
            experiment_name=f"topoformer_{hw_config['name']}"
        )
        
        try:
            result = run_experiment(config)
            results[hw_config['name']] = result
        except Exception as e:
            logger.error(f"Failed on {hw_config['name']}: {e}")
            results[hw_config['name']] = {'error': str(e)}
    
    return results


# Hardware configurations for different GPUs
HARDWARE_CONFIGS = [
    {
        'name': 'T4x4_25min',
        'device': 'cuda',
        'batch_size': 16,
        'train_samples': 5000,
        'num_epochs': 3,
        'time_limit_min': 25
    },
    {
        'name': 'L4_20min',
        'device': 'cuda',
        'batch_size': 32,
        'train_samples': 8000,
        'num_epochs': 3,
        'time_limit_min': 20
    },
    {
        'name': 'L40S_15min',
        'device': 'cuda',
        'batch_size': 64,
        'train_samples': 10000,
        'num_epochs': 2,
        'time_limit_min': 15
    },
    {
        'name': 'V100_10min',
        'device': 'cuda',
        'batch_size': 48,
        'train_samples': 10000,
        'num_epochs': 2,
        'time_limit_min': 10
    }
]


def test_with_small_samples():
    """Test with very small sample sizes (15, 20, 25, 30)"""
    sample_sizes = [15, 20, 25, 30]
    results = {}
    
    for sample_size in sample_sizes:
        logger.info(f"\nTesting with {sample_size} samples...")
        
        config = ExperimentConfig(
            model_type='topoformer',
            dataset_name='bug_localization',
            batch_size=4,
            train_samples=sample_size,
            val_samples=sample_size // 5,
            num_epochs=2,
            experiment_name=f"topoformer_test_{sample_size}samples"
        )
        
        try:
            result = run_experiment(config)
            results[f"{sample_size}_samples"] = {
                'f1': result['best_val_f1'],
                'time': result['total_training_time'],
                'memory': result['avg_memory_gb']
            }
            logger.info(f"✓ {sample_size} samples: F1={result['best_val_f1']:.4f}")
        except Exception as e:
            logger.error(f"✗ {sample_size} samples failed: {e}")
            results[f"{sample_size}_samples"] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test with small samples first
    logger.info("Running small sample tests...")
    small_results = test_with_small_samples()
    
    print("\nSmall Sample Results:")
    print("="*50)
    for size, result in small_results.items():
        print(f"{size}: {result}")
    
    # Run main experiment with 10k samples
    logger.info("\nRunning main experiment with 10k samples...")
    main_config = ExperimentConfig(
        model_type='topoformer',
        dataset_name='bug_localization',
        train_samples=10000,
        num_epochs=5,
        experiment_name='topoformer_main_10k'
    )
    
    main_result = run_experiment(main_config)
    print(f"\nMain experiment completed: Best F1 = {main_result['best_val_f1']:.4f}")
