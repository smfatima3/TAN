"""
Ablation Studies for Topoformer
Tests the contribution of each component
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import logging
import json
from tqdm import tqdm

from topoformer_complete import (
    TopoformerConfig,
    TopoformerForSequenceClassification,
    EfficientLSH,
    PersistentHomologyComputer,
    DifferentiablePersistenceLandscape,
    TopologicalAttention
)
from dataset_loader import DatasetLoader, create_data_loaders
from train_experiments import ExperimentConfig, Trainer, ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation studies"""
    # Topology settings
    use_topology: bool = True
    use_lsh: bool = True
    use_persistent_homology: bool = True
    use_landscapes: bool = True
    
    # Topology dimensions
    use_0d: bool = True
    use_1d: bool = True
    use_2d: bool = True
    
    # Hyperparameters
    k_neighbors: int = 32
    n_hashes: int = 16
    landscape_resolution: int = 50
    
    # Attention settings
    use_topological_attention: bool = True
    use_multi_scale_fusion: bool = True
    fusion_method: str = 'weighted'  # 'weighted', 'concat', 'sum'
    
    # Model architecture
    num_layers: int = 6
    num_heads: int = 12
    embed_dim: int = 768
    
    # Description
    name: str = "full_model"
    description: str = "Full Topoformer with all components"


class AblatedTopoformer(TopoformerForSequenceClassification):
    """Topoformer with ablation capabilities"""
    
    def __init__(self, base_config: TopoformerConfig, 
                 ablation_config: AblationConfig, 
                 num_labels: int):
        # Modify base config based on ablation settings
        modified_config = self._modify_config(base_config, ablation_config)
        super().__init__(modified_config, num_labels)
        
        self.ablation_config = ablation_config
        
        # Replace components based on ablation config
        self._apply_ablations()
    
    def _modify_config(self, base_config: TopoformerConfig, 
                      ablation_config: AblationConfig) -> TopoformerConfig:
        """Modify configuration based on ablation settings"""
        config = TopoformerConfig(
            vocab_size=base_config.vocab_size,
            embed_dim=ablation_config.embed_dim,
            num_layers=ablation_config.num_layers,
            num_heads=ablation_config.num_heads,
            max_seq_len=base_config.max_seq_len,
            dropout=base_config.dropout,
            k_neighbors=ablation_config.k_neighbors if ablation_config.use_topology else 0,
            n_hashes=ablation_config.n_hashes,
            hash_bits=base_config.hash_bits,
            max_homology_dim=2 if ablation_config.use_2d else (1 if ablation_config.use_1d else 0),
            landscape_resolution=ablation_config.landscape_resolution,
            use_cuda_kernel=base_config.use_cuda_kernel,
            mixed_precision=base_config.mixed_precision,
            gradient_checkpointing=base_config.gradient_checkpointing
        )
        return config
    
    def _apply_ablations(self):
        """Apply ablations by replacing components"""
        
        # If not using topology at all, replace with standard transformer layers
        if not self.ablation_config.use_topology:
            self._replace_with_standard_layers()
        
        # If not using specific topology dimensions
        elif not (self.ablation_config.use_0d and 
                  self.ablation_config.use_1d and 
                  self.ablation_config.use_2d):
            self._modify_topology_dimensions()
        
        # If not using LSH, use exact nearest neighbors
        if not self.ablation_config.use_lsh:
            self._replace_lsh_with_exact()
        
        # If not using multi-scale fusion
        if not self.ablation_config.use_multi_scale_fusion:
            self._disable_multi_scale_fusion()
    
    def _replace_with_standard_layers(self):
        """Replace Topoformer layers with standard transformer layers"""
        for i, layer in enumerate(self.layers):
            # Replace with standard multi-head attention
            standard_layer = nn.TransformerEncoderLayer(
                d_model=self.config.embed_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.embed_dim * 4,
                dropout=self.config.dropout,
                batch_first=True
            )
            self.layers[i] = standard_layer
    
    def _modify_topology_dimensions(self):
        """Modify which topology dimensions are used"""
        for layer in self.layers:
            if hasattr(layer, 'topo_attentions'):
                # Remove unused dimensions
                new_attentions = nn.ModuleList()
                
                if self.ablation_config.use_0d:
                    new_attentions.append(layer.topo_attentions[0])
                if self.ablation_config.use_1d and len(layer.topo_attentions) > 1:
                    new_attentions.append(layer.topo_attentions[1])
                if self.ablation_config.use_2d and len(layer.topo_attentions) > 2:
                    new_attentions.append(layer.topo_attentions[2])
                
                layer.topo_attentions = new_attentions
                
                # Adjust fusion weights
                n_components = 1 + len(new_attentions)  # token attention + topo attentions
                layer.fusion_weights = nn.Parameter(torch.ones(n_components))
    
    def _replace_lsh_with_exact(self):
        """Replace LSH with exact k-NN computation"""
        class ExactKNN(nn.Module):
            def forward(self, embeddings, k=32):
                batch_size, seq_len, embed_dim = embeddings.shape
                
                # Compute exact pairwise distances
                distances = torch.cdist(embeddings, embeddings, p=2)
                distances.fill_diagonal_(float('inf'))
                
                # Get k nearest neighbors
                dist_values, neighbors = torch.topk(distances, k, dim=-1, largest=False)
                
                return neighbors, dist_values
        
        for layer in self.layers:
            if hasattr(layer, 'lsh'):
                layer.lsh = ExactKNN()
    
    def _disable_multi_scale_fusion(self):
        """Use only token-level attention"""
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                # Override forward to use only token attention
                original_forward = layer.forward
                
                def new_forward(x, mask=None):
                    # Only use token attention
                    x_transposed = x.transpose(0, 1)
                    token_out, _ = layer.token_attention(
                        x_transposed, x_transposed, x_transposed, 
                        attn_mask=mask
                    )
                    token_out = token_out.transpose(0, 1)
                    
                    # Apply residual and norm
                    x = layer.norm1(x + token_out)
                    
                    # FFN
                    ffn_out = layer.ffn(x)
                    x = layer.norm2(x + ffn_out)
                    
                    return x
                
                layer.forward = new_forward


def create_ablation_configs() -> List[AblationConfig]:
    """Create all ablation configurations to test"""
    
    configs = [
        # Full model (baseline)
        AblationConfig(
            name="full_topoformer",
            description="Complete Topoformer with all components"
        ),
        
        # No topology at all
        AblationConfig(
            use_topology=False,
            use_topological_attention=False,
            name="no_topology",
            description="Standard transformer without topology"
        ),
        
        # Only 0-dimensional topology
        AblationConfig(
            use_1d=False,
            use_2d=False,
            name="0d_only",
            description="Only 0-dimensional persistence (clusters)"
        ),
        
        # 0+1 dimensional topology
        AblationConfig(
            use_2d=False,
            name="0d_1d_only",
            description="0 and 1-dimensional persistence"
        ),
        
        # Different k values
        AblationConfig(
            k_neighbors=16,
            name="k_16",
            description="Reduced neighbors (k=16)"
        ),
        
        AblationConfig(
            k_neighbors=64,
            name="k_64",
            description="Increased neighbors (k=64)"
        ),
        
        AblationConfig(
            k_neighbors=8,
            name="k_8",
            description="Minimal neighbors (k=8)"
        ),
        
        # Without LSH (exact k-NN)
        AblationConfig(
            use_lsh=False,
            name="exact_knn",
            description="Exact k-NN instead of LSH"
        ),
        
        # Different fusion methods
        AblationConfig(
            fusion_method='sum',
            name="sum_fusion",
            description="Sum fusion instead of weighted"
        ),
        
        # Fewer layers
        AblationConfig(
            num_layers=4,
            name="4_layers",
            description="Reduced to 4 layers"
        ),
        
        # Smaller embedding dimension
        AblationConfig(
            embed_dim=512,
            num_heads=8,
            name="smaller_dim",
            description="Smaller embedding dimension (512)"
        ),
        
        # Without multi-scale fusion
        AblationConfig(
            use_multi_scale_fusion=False,
            name="no_multiscale",
            description="No multi-scale fusion"
        )
    ]
    
    return configs


def run_ablation_study(dataset_name: str = 'bug_localization',
                      train_samples: int = 5000,
                      num_epochs: int = 3,
                      batch_size: int = 16) -> Dict:
    """Run complete ablation study"""
    
    logger.info("Starting ablation study...")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Training samples: {train_samples}")
    
    # Load dataset
    loader = DatasetLoader()
    if dataset_name == 'bug_localization':
        train_dataset, val_dataset, metadata = loader.load_bug_localization_dataset(
            subset_size=train_samples
        )
    elif dataset_name == 'arxiv':
        train_dataset, val_dataset, metadata = loader.load_arxiv_papers_dataset(
            subset_size=train_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size=batch_size
    )
    
    # Get ablation configurations
    ablation_configs = create_ablation_configs()
    
    # Run experiments for each configuration
    results = {}
    
    for ablation_config in ablation_configs:
        logger.info(f"\nTesting ablation: {ablation_config.name}")
        logger.info(f"Description: {ablation_config.description}")
        
        try:
            # Create base config
            base_config = TopoformerConfig()
            
            # Create ablated model
            model = AblatedTopoformer(
                base_config, 
                ablation_config, 
                num_labels=metadata['num_labels']
            )
            
            # Create experiment config
            exp_config = ExperimentConfig(
                model_type='topoformer',
                dataset_name=dataset_name,
                batch_size=batch_size,
                num_epochs=num_epochs,
                experiment_name=f"ablation_{ablation_config.name}"
            )
            
            # Create tracker and trainer
            tracker = ExperimentTracker(exp_config)
            trainer = Trainer(model, exp_config, metadata['num_labels'], tracker)
            
            # Train model
            train_results = trainer.train(train_loader, val_loader)
            
            # Save results
            results[ablation_config.name] = {
                'config': ablation_config.__dict__,
                'best_f1': train_results['best_f1'],
                'best_epoch': train_results['best_epoch'],
                'final_metrics': train_results['final_metrics'],
                'total_time': tracker.metrics['epoch_times'][-1] if tracker.metrics['epoch_times'] else 0,
                'memory_gb': np.mean(tracker.metrics['memory_usage']) if tracker.metrics['memory_usage'] else 0
            }
            
            logger.info(f"✓ {ablation_config.name}: F1={train_results['best_f1']:.4f}")
            
        except Exception as e:
            logger.error(f"✗ {ablation_config.name} failed: {e}")
            results[ablation_config.name] = {'error': str(e)}
    
    # Analyze results
    analysis = analyze_ablation_results(results)
    
    # Save results
    save_ablation_results(results, analysis, dataset_name)
    
    return results, analysis


def analyze_ablation_results(results: Dict) -> Dict:
    """Analyze ablation study results"""
    
    analysis = {
        'baseline_f1': 0,
        'component_importance': {},
        'k_sensitivity': {},
        'architecture_impact': {}
    }
    
    # Get baseline (full model) performance
    if 'full_topoformer' in results and 'error' not in results['full_topoformer']:
        baseline_f1 = results['full_topoformer']['best_f1']
        analysis['baseline_f1'] = baseline_f1
        
        # Calculate relative performance for each ablation
        for name, result in results.items():
            if 'error' not in result and name != 'full_topoformer':
                f1 = result['best_f1']
                relative_perf = (f1 / baseline_f1) * 100
                drop = baseline_f1 - f1
                
                # Categorize results
                if 'no_topology' in name:
                    analysis['component_importance']['topology'] = {
                        'drop': drop,
                        'relative': relative_perf
                    }
                elif '0d_only' in name:
                    analysis['component_importance']['higher_dim_topology'] = {
                        'drop': drop,
                        'relative': relative_perf
                    }
                elif 'k_' in name:
                    k_value = int(name.split('_')[1])
                    analysis['k_sensitivity'][k_value] = {
                        'f1': f1,
                        'drop': drop,
                        'relative': relative_perf
                    }
                elif 'layers' in name:
                    analysis['architecture_impact']['layers'] = {
                        'config': name,
                        'drop': drop,
                        'relative': relative_perf
                    }
    
    return analysis


def save_ablation_results(results: Dict, analysis: Dict, dataset_name: str):
    """Save ablation study results"""
    
    output = {
        'dataset': dataset_name,
        'results': results,
        'analysis': analysis,
        'summary': create_ablation_summary(results, analysis)
    }
    
    # Save to JSON
    with open(f'ablation_study_{dataset_name}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("ABLATION STUDY SUMMARY")
    print("="*50)
    print(output['summary'])


def create_ablation_summary(results: Dict, analysis: Dict) -> str:
    """Create human-readable summary of ablation results"""
    
    summary_lines = []
    
    # Baseline performance
    summary_lines.append(f"Baseline (Full Topoformer) F1: {analysis['baseline_f1']:.4f}")
    summary_lines.append("")
    
    # Component importance
    summary_lines.append("Component Importance:")
    if 'topology' in analysis['component_importance']:
        drop = analysis['component_importance']['topology']['drop']
        rel = analysis['component_importance']['topology']['relative']
        summary_lines.append(f"  - Without topology: -{drop:.4f} ({rel:.1f}% of baseline)")
    
    if 'higher_dim_topology' in analysis['component_importance']:
        drop = analysis['component_importance']['higher_dim_topology']['drop']
        rel = analysis['component_importance']['higher_dim_topology']['relative']
        summary_lines.append(f"  - Only 0-dim topology: -{drop:.4f} ({rel:.1f}% of baseline)")
    
    # K-sensitivity
    if analysis['k_sensitivity']:
        summary_lines.append("\nK-Neighbor Sensitivity:")
        for k in sorted(analysis['k_sensitivity'].keys()):
            f1 = analysis['k_sensitivity'][k]['f1']
            drop = analysis['k_sensitivity'][k]['drop']
            summary_lines.append(f"  - k={k}: F1={f1:.4f} (drop: -{drop:.4f})")
    
    # Architecture impact
    if analysis['architecture_impact']:
        summary_lines.append("\nArchitecture Impact:")
        for component, metrics in analysis['architecture_impact'].items():
            config = metrics['config']
            drop = metrics['drop']
            summary_lines.append(f"  - {config}: drop of -{drop:.4f}")
    
    return "\n".join(summary_lines)


def plot_ablation_results(results: Dict, save_path: str = 'ablation_results.png'):
    """Create visualization of ablation results"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    data = []
    for name, result in results.items():
        if 'error' not in result:
            data.append({
                'Configuration': name.replace('_', ' ').title(),
                'F1 Score': result['best_f1'],
                'Memory (GB)': result.get('memory_gb', 0),
                'Time (s)': result.get('total_time', 0)
            })
    
    if not data:
        logger.warning("No valid results to plot")
        return
    
    # Sort by F1 score
    data = sorted(data, key=lambda x: x['F1 Score'], reverse=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. F1 Score comparison
    ax1 = axes[0, 0]
    configs = [d['Configuration'] for d in data]
    f1_scores = [d['F1 Score'] for d in data]
    
    bars1 = ax1.barh(configs, f1_scores)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('Ablation Study: F1 Score Comparison')
    
    # Color code bars
    colors = ['#2ecc71' if 'Full' in c else '#3498db' for c in configs]
    for bar, color in zip(bars1, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (config, score) in enumerate(zip(configs, f1_scores)):
        ax1.text(score + 0.001, i, f'{score:.4f}', va='center')
    
    # 2. Relative performance drop
    ax2 = axes[0, 1]
    baseline_f1 = max(f1_scores)
    relative_drops = [(baseline_f1 - f1) * 100 / baseline_f1 for f1 in f1_scores]
    
    bars2 = ax2.barh(configs, relative_drops)
    ax2.set_xlabel('Performance Drop (%)')
    ax2.set_title('Relative Performance Drop from Baseline')
    
    # Color gradient based on drop
    for bar, drop in zip(bars2, relative_drops):
        if drop < 5:
            bar.set_color('#2ecc71')
        elif drop < 10:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#e74c3c')
    
    # 3. Memory efficiency
    ax3 = axes[1, 0]
    memory_usage = [d['Memory (GB)'] for d in data if d['Memory (GB)'] > 0]
    configs_mem = [d['Configuration'] for d in data if d['Memory (GB)'] > 0]
    
    if memory_usage:
        bars3 = ax3.barh(configs_mem, memory_usage)
        ax3.set_xlabel('Memory Usage (GB)')
        ax3.set_title('Memory Efficiency Comparison')
    
    # 4. K-sensitivity plot (if applicable)
    ax4 = axes[1, 1]
    k_results = [(int(name.split('_')[1]), result['best_f1']) 
                 for name, result in results.items() 
                 if 'k_' in name and 'error' not in result]
    
    if k_results:
        k_values, k_f1s = zip(*sorted(k_results))
        ax4.plot(k_values, k_f1s, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Neighbors (k)')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Performance vs. K-Neighbors')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Ablation results plot saved to {save_path}")


def run_minimal_ablation_test():
    """Run minimal ablation test for verification"""
    
    print("Running minimal ablation test...")
    
    # Test configurations
    test_configs = [
        AblationConfig(name="test_full", description="Full model test"),
        AblationConfig(use_topology=False, name="test_no_topo", 
                      description="No topology test"),
        AblationConfig(k_neighbors=16, name="test_k16", 
                      description="K=16 test")
    ]
    
    # Use very small dataset
    results = {}
    
    for config in test_configs:
        print(f"\nTesting: {config.name}")
        
        try:
            # Create minimal model
            base_config = TopoformerConfig(
                vocab_size=1000,
                embed_dim=128,
                num_layers=2,
                num_heads=4,
                max_seq_len=64
            )
            
            model = AblatedTopoformer(base_config, config, num_labels=3)
            
            # Test forward pass
            batch_size, seq_len = 2, 64
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
            
            results[config.name] = {
                'success': True,
                'output_shape': output['logits'].shape
            }
            print(f"✓ {config.name}: Output shape {output['logits'].shape}")
            
        except Exception as e:
            results[config.name] = {'success': False, 'error': str(e)}
            print(f"✗ {config.name}: {e}")
    
    return results


if __name__ == "__main__":
    # Run minimal test first
    print("Running minimal ablation test...")
    test_results = run_minimal_ablation_test()
    
    print("\nTest Results:")
    for name, result in test_results.items():
        print(f"{name}: {result}")
    
    # Run full ablation study
    print("\n" + "="*50)
    print("Running full ablation study...")
    print("="*50)
    
    results, analysis = run_ablation_study(
        dataset_name='bug_localization',
        train_samples=5000,
        num_epochs=3,
        batch_size=16
    )
    
    # Plot results
    plot_ablation_results(results)
    
    print("\nAblation study completed!")