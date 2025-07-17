#!/usr/bin/env python3
"""
Fix Import Issues in Topoformer Scripts
Run this script to fix all import-related errors
"""

import os
import re

def fix_baseline_models():
    """Fix the List import issue in baseline_models.py"""
    
    filename = 'baseline_models.py'
    
    if not os.path.exists(filename):
        print(f"❌ {filename} not found!")
        return False
    
    # Read the file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Fix the import line
    old_import = "from typing import Dict, Optional, Tuple"
    new_import = "from typing import Dict, Optional, Tuple, List"
    
    if old_import in content and new_import not in content:
        content = content.replace(old_import, new_import)
        
        # Write back
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed List import in {filename}")
        return True
    else:
        print(f"ℹ️  {filename} already has correct imports or different structure")
        return True


def add_missing_imports_to_all():
    """Add commonly missing imports to all Python files"""
    
    files_to_check = [
        'topoformer_complete.py',
        'dataset_loader.py',
        'baseline_models.py',
        'train_experiments.py',
        'ablation_studies.py'
    ]
    
    # Common imports that might be missing
    required_imports = {
        'typing': 'from typing import Dict, List, Tuple, Optional, Union',
        'warnings': 'import warnings\nwarnings.filterwarnings("ignore")',
        'os': 'import os',
        'sys': 'import sys',
        'json': 'import json',
        'time': 'import time',
        'numpy': 'import numpy as np',
        'torch': 'import torch'
    }
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"⚠️  {filename} not found, skipping...")
            continue
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Check which imports are missing
        missing = []
        for module, import_line in required_imports.items():
            if module == 'typing':
                # Special check for typing imports
                if 'from typing import' not in content:
                    missing.append(import_line)
                elif 'List' not in content and 'from typing' in content:
                    # Add List to existing typing import
                    content = re.sub(
                        r'from typing import ([^;\n]+)',
                        lambda m: m.group(0) + (', List' if 'List' not in m.group(1) else ''),
                        content
                    )
            elif f'import {module}' not in content:
                missing.append(import_line)
        
        if missing:
            print(f"📝 Adding missing imports to {filename}")
            # Add imports after the docstring
            lines = content.split('\n')
            insert_pos = 0
            
            # Find position after docstring
            in_docstring = False
            for i, line in enumerate(lines):
                if line.strip().startswith('"""'):
                    if not in_docstring:
                        in_docstring = True
                    else:
                        insert_pos = i + 1
                        break
            
            # Insert missing imports
            for imp in missing:
                lines.insert(insert_pos, imp)
                insert_pos += 1
            
            # Write back
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))


def create_init_file():
    """Create __init__.py to make the directory a package"""
    
    init_content = """\"\"\"
Topoformer AAAI Experiments Package
\"\"\"

# Version
__version__ = "1.0.0"

# Make imports easier
try:
    from .topoformer_complete import TopoformerConfig, TopoformerForSequenceClassification
    from .dataset_loader import DatasetLoader, create_data_loaders
    from .baseline_models import create_baseline_model
    from .train_experiments import ExperimentConfig, run_experiment
    from .ablation_studies import run_ablation_study
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")

__all__ = [
    'TopoformerConfig',
    'TopoformerForSequenceClassification',
    'DatasetLoader',
    'create_data_loaders',
    'create_baseline_model',
    'ExperimentConfig',
    'run_experiment',
    'run_ablation_study'
]
"""
    
    with open('__init__.py', 'w') as f:
        f.write(init_content)
    
    print("✅ Created __init__.py")


def test_imports():
    """Test if all imports work correctly"""
    
    print("\n🧪 Testing imports...")
    
    test_imports = [
        "from baseline_models import create_baseline_model",
        "from topoformer_complete import TopoformerConfig",
        "from dataset_loader import DatasetLoader",
        "from train_experiments import ExperimentConfig"
    ]
    
    for imp in test_imports:
        try:
            exec(imp)
            print(f"✅ {imp}")
        except Exception as e:
            print(f"❌ {imp} - Error: {e}")


def main():
    """Main function to fix all import issues"""
    
    print("🔧 Fixing Import Issues in Topoformer Scripts")
    print("=" * 50)
    
    # Fix specific known issues
    print("\n1. Fixing baseline_models.py...")
    fix_baseline_models()
    
    # Add missing imports
    print("\n2. Adding missing imports to all files...")
    add_missing_imports_to_all()
    
    # Create init file
    print("\n3. Creating __init__.py...")
    create_init_file()
    
    # Test imports
    print("\n4. Testing imports...")
    test_imports()
    
    print("\n✅ Import fixes completed!")
    print("\nYou can now run your experiments with:")
    print("  python quick_start_runner.py")
    print("  or")
    print("  python run_all_experiments.py")


if __name__ == "__main__":
    main()