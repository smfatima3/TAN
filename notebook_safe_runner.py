# Topoformer Safe Experiment Runner for Jupyter Notebook
# Run these cells one by one to debug and fix issues

# %% [markdown]
# # Safe Topoformer Experiments
# This notebook provides a safe way to run Topoformer experiments with proper error handling

# %% Cell 1: Setup and Environment Check
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Environment Check:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear any existing CUDA cache
    torch.cuda.empty_cache()
    print("✓ CUDA cache cleared")

# %% Cell 2: Import Modules with Error Handling
try:
    from train_experiments import ExperimentConfig, run_experiment
    from dataset_loader import DatasetLoader
    from topoformer_complete import TopoformerConfig, TopoformerForSequenceClassification
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all .py files are in the current directory")

# %% Cell 3: Test Data Loading
print("Testing data loading with small sample...")

loader = DatasetLoader()

try:
    # Load minimal data
    train_dataset, val_dataset, metadata = loader.load_bug_localization_dataset(
        subset_size=20  # Very small for testing
    )
    
    print(f"✓ Data loaded successfully!")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Number of labels: {metadata['num_labels']}")
    
    # Examine a sample
    sample = train_dataset[0]
    print(f"\nSample inspection:")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Max token ID: {sample['input_ids'].max().item()}")
    print(f"  Label: {sample['labels'].item()}")
    
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()

# %% Cell 4: Test Model Creation
print("Testing model creation...")

try:
    from transformers import AutoTokenizer
    
    # Get tokenizer and vocab size
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # Create minimal model config
    config = TopoformerConfig(
        vocab_size=vocab_size,
        embed_dim=256,  # Smaller for testing
        num_layers=2,   # Fewer layers
        num_heads=4,    # Fewer heads
        max_seq_len=512,  # Ensure this is large enough
        k_neighbors=16,
        mixed_precision=False,
        gradient_checkpointing=False
    )
    
    # Create model
    model = TopoformerForSequenceClassification(config, num_labels=metadata['num_labels'])
    
    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✓ Model created successfully on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones(batch_size, seq_len).to(device)
    
    with torch.no_grad():
        output = model(dummy_input, dummy_mask)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output['logits'].shape}")
    
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

# %% Cell 5: Run Small Experiment
print("Running small experiment...")

# Configure small experiment
small_config = ExperimentConfig(
    model_type='topoformer',
    dataset_name='bug_localization',
    batch_size=4,
    train_samples=30,
    val_samples=10,
    num_epochs=2,
    max_seq_length=256,
    topoformer_layers=2,
    topoformer_heads=4,
    topoformer_embed_dim=256,
    mixed_precision=False,
    gradient_accumulation_steps=1,
    experiment_name='notebook_small_test'
)

try:
    # Run experiment
    result = run_experiment(small_config)
    
    print(f"\n✓ Experiment completed successfully!")
    print(f"Best F1 Score: {result['best_val_f1']:.4f}")
    print(f"Training time: {result['total_training_time']:.2f} seconds")
    print(f"Memory usage: {result['avg_memory_gb']:.2f} GB")
    
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"❌ CUDA Error: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Try reducing batch_size to 2")
        print("2. Try reducing max_seq_length to 128")
        print("3. Try running on CPU by setting device='cpu' in config")
        print("4. Check if GPU memory is available:")
        if torch.cuda.is_available():
            print(f"   Current memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"   Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    else:
        print(f"❌ Error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

# %% Cell 6: Run CPU-Only Test (Fallback)
print("Running CPU-only test as fallback...")

cpu_config = ExperimentConfig(
    model_type='topoformer',
    dataset_name='bug_localization',
    batch_size=2,
    train_samples=20,
    val_samples=5,
    num_epochs=1,
    max_seq_length=128,
    topoformer_layers=2,
    topoformer_heads=4,
    topoformer_embed_dim=128,
    device='cpu',  # Force CPU
    mixed_precision=False,
    experiment_name='cpu_fallback_test'
)

try:
    result = run_experiment(cpu_config)
    print(f"\n✓ CPU experiment completed!")
    print(f"F1 Score: {result['best_val_f1']:.4f}")
except Exception as e:
    print(f"❌ Even CPU test failed: {e}")

# %% Cell 7: Main Experiment (if previous tests passed)
print("\nIf all previous tests passed, you can run the main experiment:")

main_config = ExperimentConfig(
    model_type='topoformer',
    dataset_name='bug_localization',
    batch_size=16,
    train_samples=10000,
    val_samples=2000,
    num_epochs=5,
    max_seq_length=256,
    mixed_precision=torch.cuda.is_available(),
    experiment_name='main_experiment_10k'
)

print(f"Main experiment configuration:")
print(f"  Dataset: {main_config.dataset_name}")
print(f"  Training samples: {main_config.train_samples}")
print(f"  Batch size: {main_config.batch_size}")
print(f"  Epochs: {main_config.num_epochs}")
print(f"  Device: {main_config.device}")

# Uncomment to run:
# result = run_experiment(main_config)
# print(f"Main experiment F1: {result['best_val_f1']:.4f}")