#!/usr/bin/env python3
"""
Run All Topoformer Experiments
This script runs the complete experimental pipeline for the AAAI paper
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'topoformer_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }
    
    optional_packages = {
        'gudhi': 'GUDHI (for exact topology)',
        'ripser': 'Ripser (for topology)',
        'wandb': 'Weights & Biases (for tracking)'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name} is installed")
        except ImportError:
            missing_required.append(name)
            logger.error(f"✗ {name} is NOT installed")
    
    # Check optional packages
    for package, name in optional_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name} is installed (optional)")
        except ImportError:
            missing_optional.append(name)
            logger.warning(f"⚠ {name} is NOT installed (optional)")
    
    if missing_required:
        logger.error(f"\nMissing required packages: {', '.join(missing_required)}")
        logger.error("Please install with: pip install torch transformers datasets scikit-learn numpy matplotlib tqdm")
        return False
    
    if missing_optional:
        logger.info(f"\nOptional packages not installed: {', '.join(missing_optional)}")
        logger.info("For full functionality, install with: pip install gudhi ripser wandb")
    
    return True


def run_small_tests():
    """Run tests with small sample sizes"""
    logger.info("\n" + "="*50)
    logger.info("PHASE 1: Small Sample Tests (15, 20, 25, 30)")
    logger.info("="*50)
    
    from train_experiments import test_with_small_samples
    
    try:
        results = test_with_small_samples()
        logger.info("Small sample tests completed successfully")
        return results
    except Exception as e:
        logger.error(f"Small sample tests failed: {e}")
        return None


def run_main_experiment(dataset='bug_localization', samples=10000):
    """Run main experiment with specified samples"""
    logger.info("\n" + "="*50)
    logger.info(f"PHASE 2: Main Experiment ({samples} samples)")
    logger.info("="*50)
    
    from train_experiments import ExperimentConfig, run_experiment
    
    config = ExperimentConfig(
        model_type='topoformer',
        dataset_name=dataset,
        train_samples=samples,
        num_epochs=5,
        batch_size=16,
        experiment_name=f'topoformer_main_{samples}'
    )
    
    try:
        result = run_experiment(config)
        logger.info(f"Main experiment completed: F1={result['best_val_f1']:.4f}")
        return result
    except Exception as e:
        logger.error(f"Main experiment failed: {e}")
        return None


def run_baseline_comparisons(dataset='bug_localization', samples=10000):
    """Run baseline model comparisons"""
    logger.info("\n" + "="*50)
    logger.info("PHASE 3: Baseline Comparisons")
    logger.info("="*50)
    
    from train_experiments import ExperimentConfig, run_experiment
    
    baselines = ['bert', 'codebert', 'longformer']
    results = {}
    
    for baseline in baselines:
        logger.info(f"\nTesting {baseline}...")
        
        config = ExperimentConfig(
            model_type=baseline,
            dataset_name=dataset,
            train_samples=samples,
            num_epochs=3,
            batch_size=16,
            experiment_name=f'{baseline}_comparison'
        )
        
        try:
            result = run_experiment(config)
            results[baseline] = result
            logger.info(f"✓ {baseline}: F1={result['best_val_f1']:.4f}")
        except Exception as e:
            logger.error(f"✗ {baseline} failed: {e}")
            results[baseline] = {'error': str(e)}
    
    return results


def run_ablation_studies(dataset='bug_localization', samples=5000):
    """Run ablation studies"""
    logger.info("\n" + "="*50)
    logger.info("PHASE 4: Ablation Studies")
    logger.info("="*50)
    
    from ablation_studies import run_ablation_study
    
    try:
        results, analysis = run_ablation_study(
            dataset_name=dataset,
            train_samples=samples,
            num_epochs=3,
            batch_size=16
        )
        logger.info("Ablation studies completed successfully")
        return results, analysis
    except Exception as e:
        logger.error(f"Ablation studies failed: {e}")
        return None, None


def run_hardware_benchmarks():
    """Run hardware-specific benchmarks"""
    logger.info("\n" + "="*50)
    logger.info("PHASE 5: Hardware Benchmarks")
    logger.info("="*50)
    
    import torch
    
    if not torch.cuda.is_available():
        logger.warning("No GPU available, skipping hardware benchmarks")
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Current GPU: {gpu_name}")
    
    # Determine appropriate configuration based on GPU
    from train_experiments import ExperimentConfig, run_experiment
    
    # Simplified config based on available GPU
    config = ExperimentConfig(
        model_type='topoformer',
        dataset_name='bug_localization',
        train_samples=5000,
        num_epochs=2,
        batch_size=16,
        experiment_name=f'hardware_benchmark_{gpu_name.replace(" ", "_")}'
    )
    
    try:
        result = run_experiment(config)
        logger.info(f"Hardware benchmark completed on {gpu_name}")
        return result
    except Exception as e:
        logger.error(f"Hardware benchmark failed: {e}")
        return None


def generate_report(all_results):
    """Generate final report"""
    logger.info("\n" + "="*50)
    logger.info("GENERATING FINAL REPORT")
    logger.info("="*50)
    
    report = []
    report.append("# Topoformer AAAI Experiment Results\n")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Small sample results
    if all_results.get('small_samples'):
        report.append("## Small Sample Tests\n")
        for size, result in all_results['small_samples'].items():
            if isinstance(result, dict) and 'f1' in result:
                report.append(f"- {size} samples: F1={result['f1']:.4f}\n")
    
    # Main experiment results
    if all_results.get('main_experiment'):
        report.append("\n## Main Experiment Results\n")
        main_res = all_results['main_experiment']
        report.append(f"- Best F1 Score: {main_res.get('best_val_f1', 0):.4f}\n")
        report.append(f"- Training Time: {main_res.get('total_training_time', 0)/60:.1f} minutes\n")
        report.append(f"- Memory Usage: {main_res.get('avg_memory_gb', 0):.2f} GB\n")
    
    # Baseline comparisons
    if all_results.get('baselines'):
        report.append("\n## Baseline Comparisons\n")
        for model, result in all_results['baselines'].items():
            if isinstance(result, dict) and 'best_val_f1' in result:
                report.append(f"- {model}: F1={result['best_val_f1']:.4f}\n")
    
    # Ablation studies
    if all_results.get('ablations'):
        report.append("\n## Ablation Study Results\n")
        ablation_analysis = all_results.get('ablation_analysis', {})
        if ablation_analysis.get('baseline_f1'):
            report.append(f"- Baseline F1: {ablation_analysis['baseline_f1']:.4f}\n")
            if ablation_analysis.get('component_importance'):
                for comp, metrics in ablation_analysis['component_importance'].items():
                    report.append(f"- Without {comp}: {metrics['drop']:.4f} drop\n")
    
    # Save report
    report_text = ''.join(report)
    with open('topoformer_experiment_report.md', 'w') as f:
        f.write(report_text)
    
    logger.info("Report saved to topoformer_experiment_report.md")
    print("\n" + report_text)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run Topoformer experiments')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'small', 'main', 'baselines', 'ablations', 'hardware'],
                       help='Which phase to run')
    parser.add_argument('--dataset', type=str, default='bug_localization',
                       choices=['bug_localization', 'arxiv', 'multi_eurlex', 'wikipedia'],
                       help='Dataset to use')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer samples')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.samples = 1000
        logger.info("Running in quick mode with reduced samples")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Please install missing packages.")
        return 1
    
    # Initialize results storage
    all_results = {}
    
    # Start timer
    start_time = time.time()
    
    try:
        if args.phase == 'all' or args.phase == 'small':
            all_results['small_samples'] = run_small_tests()
        
        if args.phase == 'all' or args.phase == 'main':
            all_results['main_experiment'] = run_main_experiment(
                dataset=args.dataset, 
                samples=args.samples
            )
        
        if args.phase == 'all' or args.phase == 'baselines':
            all_results['baselines'] = run_baseline_comparisons(
                dataset=args.dataset,
                samples=args.samples
            )
        
        if args.phase == 'all' or args.phase == 'ablations':
            ablation_results, ablation_analysis = run_ablation_studies(
                dataset=args.dataset,
                samples=args.samples // 2  # Use fewer samples for ablations
            )
            all_results['ablations'] = ablation_results
            all_results['ablation_analysis'] = ablation_analysis
        
        if args.phase == 'all' or args.phase == 'hardware':
            all_results['hardware'] = run_hardware_benchmarks()
        
        # Generate final report
        generate_report(all_results)
        
        # Total time
        total_time = time.time() - start_time
        logger.info(f"\nTotal execution time: {total_time/60:.1f} minutes")
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())