"""
Topoformer Training Pipeline
Handles dataset loading, training, evaluation, and result visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BugDataset(Dataset):
    """Dataset for bug classification"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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
        
        # Extract labels
        labels = item['labels']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': {
                'component': torch.tensor(labels['component'], dtype=torch.long),
                'sub_component': torch.tensor(labels['sub_component'], dtype=torch.long),
                'bug_type': torch.tensor(labels['bug_type'], dtype=torch.long),
                'severity': torch.tensor(labels['severity'], dtype=torch.long)
            },
            'bug_id': item['bug_id']
        }


class TopoformerTrainer:
    """Training class for Topoformer"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 save_dir: str = "./results"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        total_steps = len(train_loader) * config.get('num_epochs', 5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = {
            'component': [],
            'subcomponent': [],
            'bugtype': [],
            'severity': []
        }
        all_labels = {
            'component': [],
            'subcomponent': [],
            'bugtype': [],
            'severity': []
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Get predictions
            with torch.no_grad():
                all_predictions['component'].extend(
                    outputs['component_logits'].argmax(dim=-1).cpu().numpy()
                )
                all_predictions['subcomponent'].extend(
                    outputs['subcomponent_logits'].argmax(dim=-1).cpu().numpy()
                )
                all_predictions['bugtype'].extend(
                    outputs['bugtype_logits'].argmax(dim=-1).cpu().numpy()
                )
                all_predictions['severity'].extend(
                    outputs['severity_logits'].argmax(dim=-1).cpu().numpy()
                )
                
                all_labels['component'].extend(labels['component'].cpu().numpy())
                all_labels['subcomponent'].extend(labels['sub_component'].cpu().numpy())
                all_labels['bugtype'].extend(labels['bug_type'].cpu().numpy())
                all_labels['severity'].extend(labels['severity'].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_labels)
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'metrics': metrics
        }
    
    def evaluate(self) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = {
            'component': [],
            'subcomponent': [],
            'bugtype': [],
            'severity': []
        }
        all_labels = {
            'component': [],
            'subcomponent': [],
            'bugtype': [],
            'severity': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                
                # Get predictions
                all_predictions['component'].extend(
                    outputs['component_logits'].argmax(dim=-1).cpu().numpy()
                )
                all_predictions['subcomponent'].extend(
                    outputs['subcomponent_logits'].argmax(dim=-1).cpu().numpy()
                )
                all_predictions['bugtype'].extend(
                    outputs['bugtype_logits'].argmax(dim=-1).cpu().numpy()
                )
                all_predictions['severity'].extend(
                    outputs['severity_logits'].argmax(dim=-1).cpu().numpy()
                )
                
                all_labels['component'].extend(labels['component'].cpu().numpy())
                all_labels['subcomponent'].extend(labels['sub_component'].cpu().numpy())
                all_labels['bugtype'].extend(labels['bug_type'].cpu().numpy())
                all_labels['severity'].extend(labels['severity'].cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_labels)
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'metrics': metrics
        }
    
    def calculate_metrics(self, predictions: Dict, labels: Dict) -> Dict:
        """Calculate F1 scores and accuracies"""
        metrics = {}
        
        for task in ['component', 'subcomponent', 'bugtype', 'severity']:
            pred = predictions[task]
            true = labels[task]
            
            # Use macro F1 for multi-class classification
            f1 = f1_score(true, pred, average='macro')
            acc = accuracy_score(true, pred)
            
            metrics[f"{task}_f1"] = f1
            metrics[f"{task}_acc"] = acc
        
        # Overall F1 (average across all tasks)
        metrics['overall_f1'] = np.mean([
            metrics['component_f1'],
            metrics['subcomponent_f1'],
            metrics['bugtype_f1'],
            metrics['severity_f1']
        ])
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict:
        """Complete training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        best_val_f1 = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_results = self.train_epoch(epoch + 1)
            self.history['train_loss'].append(train_results['loss'])
            self.history['train_metrics'].append(train_results['metrics'])
            
            # Evaluate
            val_results = self.evaluate()
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_metrics'].append(val_results['metrics'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log results
            logger.info(f"Train Loss: {train_results['loss']:.4f}")
            logger.info(f"Train Overall F1: {train_results['metrics']['overall_f1']:.4f}")
            logger.info(f"Val Loss: {val_results['loss']:.4f}")
            logger.info(f"Val Overall F1: {val_results['metrics']['overall_f1']:.4f}")
            
            # Save best model
            if val_results['metrics']['overall_f1'] > best_val_f1:
                best_val_f1 = val_results['metrics']['overall_f1']
                best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt', epoch, val_results['metrics'])
                logger.info(f"New best model saved! F1: {best_val_f1:.4f}")
        
        # Save final model and results
        self.save_checkpoint('final_model.pt', num_epochs, val_results['metrics'])
        self.save_results()
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
        
        return {
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'history': self.history
        }
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def save_results(self):
        """Save training history and create visualizations"""
        # Save history to JSON
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Create visualizations
        self.plot_training_curves()
        self.plot_metrics_comparison()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 curves
        train_f1 = [m['overall_f1'] for m in self.history['train_metrics']]
        val_f1 = [m['overall_f1'] for m in self.history['val_metrics']]
        
        ax2.plot(epochs, train_f1, 'b-', label='Train F1')
        ax2.plot(epochs, val_f1, 'r-', label='Val F1')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Training and Validation F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
    def plot_metrics_comparison(self):
        """Plot comparison of metrics across tasks"""
        # Get final metrics
        final_metrics = self.history['val_metrics'][-1]
        
        tasks = ['component', 'subcomponent', 'bugtype', 'severity']
        f1_scores = [final_metrics[f"{task}_f1"] for task in tasks]
        accuracies = [final_metrics[f"{task}_acc"] for task in tasks]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # F1 scores
        bars1 = ax1.bar(tasks, f1_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Scores by Task')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Accuracies
        bars2 = ax2.bar(tasks, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracies by Task')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()


def load_synthetic_dataset(data_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load synthetic bug dataset"""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train_data = data['train']
    val_data = data['validation']
    test_data = data['test']
    
    logger.info(f"Dataset loaded:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Validation: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data


def create_data_loaders(train_data: List[Dict], 
                       val_data: List[Dict],
                       tokenizer,
                       batch_size: int = 16,
                       max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation"""
    
    train_dataset = BugDataset(train_data, tokenizer, max_length)
    val_dataset = BugDataset(val_data, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader,
                  device: torch.device,
                  save_dir: str) -> Dict:
    """Comprehensive evaluation on test set"""
    model.eval()
    
    all_predictions = {
        'component': [],
        'subcomponent': [],
        'bugtype': [],
        'severity': []
    }
    all_labels = {
        'component': [],
        'subcomponent': [],
        'bugtype': [],
        'severity': []
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            outputs = model(input_ids, attention_mask)
            
            # Get predictions
            all_predictions['component'].extend(
                outputs['component_logits'].argmax(dim=-1).cpu().numpy()
            )
            all_predictions['subcomponent'].extend(
                outputs['subcomponent_logits'].argmax(dim=-1).cpu().numpy()
            )
            all_predictions['bugtype'].extend(
                outputs['bugtype_logits'].argmax(dim=-1).cpu().numpy()
            )
            all_predictions['severity'].extend(
                outputs['severity_logits'].argmax(dim=-1).cpu().numpy()
            )
            
            all_labels['component'].extend(labels['component'].cpu().numpy())
            all_labels['subcomponent'].extend(labels['sub_component'].cpu().numpy())
            all_labels['bugtype'].extend(labels['bug_type'].cpu().numpy())
            all_labels['severity'].extend(labels['severity'].cpu().numpy())
    
    # Calculate detailed metrics
    results = {}
    
    for task in ['component', 'subcomponent', 'bugtype', 'severity']:
        pred = all_predictions[task]
        true = all_labels[task]
        
        # Classification report
        report = classification_report(true, pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true, pred)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {task.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{task}.png'), dpi=300)
        plt.close()
        
        results[task] = {
            'f1_score': f1_score(true, pred, average='macro'),
            'accuracy': accuracy_score(true, pred),
            'classification_report': report
        }
    
    # Overall metrics
    results['overall_f1'] = np.mean([
        results['component']['f1_score'],
        results['subcomponent']['f1_score'],
        results['bugtype']['f1_score'],
        results['severity']['f1_score']
    ])
    
    # Save results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTest Results:")
    logger.info(f"Overall F1: {results['overall_f1']:.4f}")
    for task in ['component', 'subcomponent', 'bugtype', 'severity']:
        logger.info(f"{task.capitalize()} - F1: {results[task]['f1_score']:.4f}, "
                   f"Acc: {results[task]['accuracy']:.4f}")
    
    return results