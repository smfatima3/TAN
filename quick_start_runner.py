#!/usr/bin/env python3
"""
Quick Start Runner for Topoformer Experiments
Handles common errors and provides a simple interface
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Handle import errors gracefully
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print("✓ FlashAttention is available")
except ImportError:
    print("⚠ FlashAttention not available (this is OK, using standard attention)")

GUDHI_AVAILABLE = False
try:
    import gudhi
    GUDHI_AVAILABLE = True
    print("✓ GUDHI is available")
except ImportError:
    print("⚠ GUDHI not available (using approximation)")

print("\nSetting up environment...")

# Core imports
import torch
import numpy as np
from typing import Dict, List, Optional
import json
import time
from datetime import datetime

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Import our modules with error handling
try:
    from topoformer_complete import TopoformerConfig, TopoformerForSequenceClassification
    from dataset_loader import DatasetLoader, create_data_loaders
    from train_experiments import ExperimentConfig, Trainer, ExperimentTracker
    print("✓ Topoformer modules loaded")
except ImportError as e:
    print(f"✗ Error loading modules: {e}")
    print("Make sure all script files are in the same directory")
    sys.exit(1)


def run_quick_test():
    """Run a quick test with minimal data"""
    print("\n" + "="*50)
    print("Running Quick Test (30 samples)")
    print("="*50)
    
    try:
        # Create minimal config
        config = ExperimentConfig(
            model_type='topoformer',
            dataset_name='bug_localization',
            batch_size=4,
            train_samples=30,
            val_samples=10,
            num_epochs=2,
            mixed_precision=False,  # Disable for compatibility
            experiment_name='quick_test'
        )
        
        # Load data
        loader = DatasetLoader()
        train_dataset, val_dataset, metadata = loader.load_bug_localization_dataset(
            subset_size=config.train_samples
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, 
            batch_size=config.batch_size
        )
        
        # Create model
        topo_config = TopoformerConfig(
            vocab_size=30000,
            embed_dim=256,  # Smaller for quick test
            num_layers=2,
            num_heads=4,
            mixed_precision=False
        )
        model = TopoformerForSequenceClassification(topo_config, metadata['num_labels'])
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train
        tracker = ExperimentTracker(config)
        trainer = Trainer(model, config, metadata['num_labels'], tracker)
        
        print("\nTraining...")
        results = trainer.train(train_loader, val_loader)
        
        print(f"\n✓ Quick test completed!")
        print(f"Best F1: {results['best_f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_main_experiments():
    """Run main experiments with proper error handling"""
    print("\n" + "="*50)
    print("Running Main Experiments")
    print("="*50)
    
    # Test configurations based on available time
    configs = [
        {
            'name': 'small_test',
            'samples': 100,
            'epochs': 2,
            'batch_size': 8
        },
        {
            'name': 'medium_test',
            'samples': 1000,
            'epochs': 3,
            'batch_size': 16
        },
        {
            'name': 'full_test',
            'samples': 10000,
            'epochs': 5,
            'batch_size': 16
        }
    ]
    
    results = {}
    
    for cfg in configs:
        print(f"\nRunning {cfg['name']} ({cfg['samples']} samples)...")
        
        try:
            config = ExperimentConfig(
                model_type='topoformer',
                dataset_name='bug_localization',
                batch_size=cfg['batch_size'],
                train_samples=cfg['samples'],
                num_epochs=cfg['epochs'],
                mixed_precision=torch.cuda.is_available(),  # Only use on GPU
                experiment_name=cfg['name']
            )
            
            # Load dataset
            loader = DatasetLoader()
            train_dataset, val_dataset, metadata = loader.load_bug_localization_dataset(
                subset_size=config.train_samples
            )
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                train_dataset, val_dataset,
                batch_size=config.batch_size
            )
            
            # Create model
            topo_config = TopoformerConfig(
                vocab_size=30000,
                embed_dim=768,
                num_layers=6,
                num_heads=12,
                mixed_precision=config.mixed_precision
            )
            model = TopoformerForSequenceClassification(topo_config, metadata['num_labels'])
            
            # Train
            tracker = ExperimentTracker(config)
            trainer = Trainer(model, config, metadata['num_labels'], tracker)
            
            train_results = trainer.train(train_loader, val_loader)
            
            results[cfg['name']] = {
                'samples': cfg['samples'],
                'best_f1': train_results['best_f1'],
                'time': sum(tracker.metrics['epoch_times']),
                'memory': np.mean(tracker.metrics['memory_usage']) if tracker.metrics['memory_usage'] else 0
            }
            
            print(f"✓ {cfg['name']}: F1={train_results['best_f1']:.4f}")
            
            # Save checkpoint
            tracker.save_results()
            
        except Exception as e:
            print(f"✗ {cfg['name']} failed: {e}")
            results[cfg['name']] = {'error': str(e)}
    
    return results


def test_baselines():
    """Test baseline models with error handling"""
    print("\n" + "="*50)
    print("Testing Baseline Models")
    print("="*50)
    
    # Only test models that don't require FlashAttention
    from baseline_models import create_baseline_model
    
    baselines_to_test = ['bert', 'codebert']  # Skip FlashAttention-dependent models
    
    for baseline in baselines_to_test:
        print(f"\nTesting {baseline}...")
        try:
            model = create_baseline_model(baseline, num_labels=5, device=str(device))
            
            # Test forward pass
            dummy_input = torch.randint(0, 1000, (2, 128)).to(device)
            dummy_mask = torch.ones(2, 128).to(device)
            
            _ = model.model(input_ids=dummy_input, attention_mask=dummy_mask)
            
            print(f"✓ {baseline} works correctly")
            
        except Exception as e:
            print(f"✗ {baseline} failed: {e}")


def main():
    """Main execution with menu"""
    print("\nTopoformer Experiment Runner")
    print("="*50)
    
    while True:
        print("\nSelect an option:")
        print("1. Run quick test (30 samples)")
        print("2. Run small experiment (100 samples)")
        print("3. Run medium experiment (1000 samples)")
        print("4. Run full experiment (10000 samples)")
        print("5. Test baseline models")
        print("6. Run all tests")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            run_quick_test()
        elif choice == '2':
            results = run_main_experiments()
            print(f"\nResults: {results.get('small_test', 'Not run')}")
        elif choice == '3':
            results = run_main_experiments()
            print(f"\nResults: {results.get('medium_test', 'Not run')}")
        elif choice == '4':
            results = run_main_experiments()
            print(f"\nResults: {results.get('full_test', 'Not run')}")
        elif choice == '5':
            test_baselines()
        elif choice == '6':
            print("\nRunning all tests...")
            run_quick_test()
            test_baselines()
            results = run_main_experiments()
            
            # Save final results
            with open('experiment_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nAll results saved to experiment_results.json")
        else:
            print("Invalid choice. Please try again.")
    
    print("\nExperiment runner finished.")


if __name__ == "__main__":
    # Run quick test first to verify setup
    if run_quick_test():
        print("\n✓ Setup verified. You can now run main experiments.")
        main()
    else:
        print("\n✗ Setup verification failed. Please check error messages above.")
        print("\nCommon fixes:")
        print("1. Make sure all .py files are in the same directory")
        print("2. Install missing packages: pip install torch transformers datasets scikit-learn")
        print("3. If CUDA errors occur, try: export CUDA_VISIBLE_DEVICES=0")