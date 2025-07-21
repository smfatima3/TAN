#!/usr/bin/env python3
"""
Safe ArXiv Experiment Script
Handles all tokenization and vocabulary issues
"""

import os
import torch
import numpy as np
from train_experiments import ExperimentConfig, run_experiment
from dataset_loader import DatasetLoader
from transformers import AutoTokenizer

# Set environment for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_arxiv_dataset():
    """Test ArXiv dataset loading and check for issues"""
    print("Testing ArXiv dataset...")
    
    loader = DatasetLoader()
    train_dataset, val_dataset, metadata = loader.load_arxiv_papers_dataset(
        subset_size=100  # Small sample for testing
    )
    
    print(f"Dataset loaded successfully!")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print(f"  - Number of labels: {metadata['num_labels']}")
    print(f"  - Vocab size: {metadata.get('vocab_size', 'Not specified')}")
    
    # Check a sample
    sample = train_dataset[0]
    print(f"\nSample check:")
    print(f"  - Input shape: {sample['input_ids'].shape}")
    print(f"  - Max token ID: {sample['input_ids'].max().item()}")
    print(f"  - Min token ID: {sample['input_ids'].min().item()}")
    print(f"  - Label: {sample['labels'].item()}")
    
    # Verify vocab size
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    actual_vocab_size = len(tokenizer)
    print(f"\nTokenizer check:")
    print(f"  - Actual vocab size: {actual_vocab_size}")
    
    if sample['input_ids'].max().item() >= actual_vocab_size:
        print(f"  ⚠️ WARNING: Token IDs exceed vocab size!")
        print(f"  Max token ID: {sample['input_ids'].max().item()}")
        print(f"  Vocab size: {actual_vocab_size}")
        return False
    else:
        print(f"  ✓ All token IDs are within vocab range")
        return True


def run_safe_arxiv_experiment(sample_size=5000):
    """Run ArXiv experiment with all safety checks"""
    
    print(f"\nRunning ArXiv experiment with {sample_size} samples")
    print("="*50)
    
    # First test the dataset
    if not test_arxiv_dataset():
        print("Dataset test failed! Aborting...")
        return None
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create configuration
    config = ExperimentConfig(
        model_type='topoformer',
        dataset_name='arxiv',
        batch_size=16,
        train_samples=sample_size,
        val_samples=int(sample_size * 0.2),
        num_epochs=5,
        learning_rate=5e-5,
        max_seq_length=256,  # Reduced for safety
        mixed_precision=False,  # Disable to avoid NaN
        gradient_accumulation_steps=1,
        topoformer_layers=4,  # Smaller model
        topoformer_embed_dim=512,
        topoformer_heads=8,
        k_neighbors=16,
        experiment_name=f'arxiv_safe_{sample_size}'
    )
    
    print("\nConfiguration:")
    print(f"  - Model: {config.model_type}")
    print(f"  - Dataset: {config.dataset_name}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Max seq length: {config.max_seq_length}")
    print(f"  - Layers: {config.topoformer_layers}")
    print(f"  - Embed dim: {config.topoformer_embed_dim}")
    
    try:
        # Run experiment
        result = run_experiment(config)
        
        print(f"\n✓ Experiment completed successfully!")
        print(f"Best F1 Score: {result['best_val_f1']:.4f}")
        print(f"Final F1 Score: {result.get('final_metrics', {}).get('val_f1', 0):.4f}")
        print(f"Training time: {result['total_training_time']:.2f} seconds")
        print(f"Memory usage: {result['avg_memory_gb']:.2f} GB")
        
        # Check if results are reasonable
        if result['best_val_f1'] < 0.1:
            print("\n⚠️ WARNING: F1 score is very low. Possible issues:")
            print("  - Learning rate might be too low/high")
            print("  - Model might need more epochs")
            print("  - Check loss values in training logs")
        elif result['best_val_f1'] > 0.5:
            print("\n✓ Good results! Model is learning properly.")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_baseline_comparison():
    """Run BERT baseline for comparison"""
    print("\nRunning BERT baseline for comparison...")
    
    config = ExperimentConfig(
        model_type='bert',
        dataset_name='arxiv',
        batch_size=16,
        train_samples=5000,
        num_epochs=3,
        learning_rate=5e-5,
        experiment_name='bert_arxiv_baseline'
    )
    
    try:
        result = run_experiment(config)
        print(f"BERT Baseline F1: {result['best_val_f1']:.4f}")
        return result
    except Exception as e:
        print(f"Baseline failed: {e}")
        return None


def analyze_results(topoformer_result, baseline_result=None):
    """Analyze and compare results"""
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)
    
    if topoformer_result:
        print(f"\nTopoformer Results:")
        print(f"  - Best F1: {topoformer_result['best_val_f1']:.4f}")
        print(f"  - Training time: {topoformer_result['total_training_time']/60:.1f} minutes")
        print(f"  - Memory usage: {topoformer_result['avg_memory_gb']:.2f} GB")
        
        # Expected F1 for 11-class ArXiv
        expected_random = 1/11  # ~0.091
        expected_good = 0.6
        
        if topoformer_result['best_val_f1'] < expected_random * 1.5:
            print(f"\n✗ Model is barely better than random ({expected_random:.3f})")
            print("  Recommendations:")
            print("  - Increase learning rate to 1e-4")
            print("  - Train for more epochs (10+)")
            print("  - Check if loss is decreasing")
        elif topoformer_result['best_val_f1'] < 0.3:
            print(f"\n⚠️ Model is learning but performance is low")
            print("  Recommendations:")
            print("  - Try different learning rates (1e-4, 2e-4)")
            print("  - Use larger model (6 layers, 768 dim)")
            print("  - Train for more epochs")
        else:
            print(f"\n✓ Good performance achieved!")
    
    if baseline_result:
        print(f"\nBERT Baseline Results:")
        print(f"  - Best F1: {baseline_result['best_val_f1']:.4f}")
        
        if topoformer_result and baseline_result:
            improvement = ((topoformer_result['best_val_f1'] - baseline_result['best_val_f1']) / 
                          baseline_result['best_val_f1'] * 100)
            print(f"\nTopoformer vs BERT: {improvement:+.1f}%")


def main():
    """Main execution"""
    print("SAFE ARXIV EXPERIMENT RUNNER")
    print("="*50)
    
    # Run experiments
    topo_result = run_safe_arxiv_experiment(sample_size=5000)
    
    if topo_result and topo_result['best_val_f1'] > 0.1:
        # Only run baseline if Topoformer worked
        baseline_result = run_baseline_comparison()
        analyze_results(topo_result, baseline_result)
    else:
        analyze_results(topo_result)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()