"""
Synthetic Bug Dataset Generator using DeepSeek-Coder-V2
Generates 25K realistic bug reports with hierarchical structure
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
import random
import re
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BugReport:
    """Structure for a bug report with hierarchical information"""
    bug_id: str
    title: str
    description: str
    code_snippet: str
    stack_trace: Optional[str]
    severity: str  # Critical, High, Medium, Low
    priority: str  # P0, P1, P2, P3
    component: str  # Frontend, Backend, Database, API, etc.
    sub_component: str  # More specific component
    bug_type: str  # Logic Error, Syntax Error, Performance, Security, etc.
    language: str  # Python, JavaScript, Java, etc.
    file_path: str  # Hierarchical file path
    line_numbers: List[int]
    functions_affected: List[str]
    classes_affected: List[str]
    dependencies: List[str]  # Related files/modules
    created_date: str
    reporter: str
    assignee: Optional[str]
    status: str  # Open, In Progress, Resolved, Closed
    resolution: Optional[str]  # Fixed, Won't Fix, Duplicate, etc.
    labels: List[str]  # Additional tags
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'bug_id': self.bug_id,
            'title': self.title,
            'description': self.description,
            'code_snippet': self.code_snippet,
            'stack_trace': self.stack_trace,
            'severity': self.severity,
            'priority': self.priority,
            'component': self.component,
            'sub_component': self.sub_component,
            'bug_type': self.bug_type,
            'language': self.language,
            'file_path': self.file_path,
            'line_numbers': self.line_numbers,
            'functions_affected': self.functions_affected,
            'classes_affected': self.classes_affected,
            'dependencies': self.dependencies,
            'created_date': self.created_date,
            'reporter': self.reporter,
            'assignee': self.assignee,
            'status': self.status,
            'resolution': self.resolution,
            'labels': self.labels
        }


class SyntheticBugGenerator:
    """Generate synthetic bug reports with realistic patterns"""
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        
        # If using LLM for generation
        if use_llm:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                logger.info("Loading DeepSeek-Coder model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-Coder-V2-Instruct", 
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/DeepSeek-Coder-V2-Instruct", 
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load LLM: {e}. Using template-based generation.")
                self.use_llm = False
        
        # Templates and options for synthetic generation
        self.setup_templates()
        
    def setup_templates(self):
        """Setup templates for bug generation"""
        
        # Bug type templates
        self.bug_templates = {
            'Logic Error': [
                "Function {function} returns incorrect value when {condition}",
                "Incorrect calculation in {function} for edge case: {edge_case}",
                "Logic flaw in {component} causes {symptom}"
            ],
            'Null Pointer': [
                "NullPointerException in {class}.{method} when {condition}",
                "Unhandled null value in {function} at line {line}",
                "Missing null check in {component} causes crash"
            ],
            'Performance': [
                "Slow query in {function} takes {time}ms for {operation}",
                "Memory leak in {component} when processing {data_type}",
                "High CPU usage in {function} due to {reason}"
            ],
            'Security': [
                "SQL injection vulnerability in {function} with parameter {param}",
                "XSS vulnerability in {component} when displaying {data}",
                "Authentication bypass in {endpoint} allows {attack}"
            ],
            'Concurrency': [
                "Race condition in {class} between {thread1} and {thread2}",
                "Deadlock occurs when {condition} in {component}",
                "Thread safety issue in {function} causes {symptom}"
            ]
        }
        
        # Component hierarchy
        self.components = {
            'Frontend': ['UI Components', 'Forms', 'Navigation', 'State Management'],
            'Backend': ['API Endpoints', 'Business Logic', 'Data Processing', 'Authentication'],
            'Database': ['Queries', 'Migrations', 'Indexes', 'Connections'],
            'Infrastructure': ['Deployment', 'Monitoring', 'Logging', 'Configuration'],
            'Mobile': ['iOS', 'Android', 'React Native', 'Flutter']
        }
        
        # Programming languages and their typical issues
        self.languages = {
            'Python': ['TypeError', 'ValueError', 'ImportError', 'IndentationError'],
            'JavaScript': ['TypeError', 'ReferenceError', 'SyntaxError', 'Promise rejection'],
            'Java': ['NullPointerException', 'ClassCastException', 'ArrayIndexOutOfBounds'],
            'Go': ['nil pointer dereference', 'goroutine leak', 'channel deadlock'],
            'Rust': ['borrow checker error', 'lifetime issue', 'trait bound not satisfied']
        }
        
        # Severity mapping
        self.severity_priority_map = {
            'Critical': 'P0',
            'High': 'P1',
            'Medium': 'P2',
            'Low': 'P3'
        }
        
        # Sample developers
        self.developers = [
            'alice.smith', 'bob.jones', 'carol.white', 'david.brown',
            'emma.davis', 'frank.miller', 'grace.wilson', 'henry.moore'
        ]
        
    def generate_bug_with_llm(self, bug_type: str, component: str, 
                             language: str) -> Tuple[str, str, str]:
        """Generate bug details using LLM"""
        prompt = f"""Generate a realistic bug report for a {bug_type} in the {component} component written in {language}.

Include:
1. A concise title
2. A detailed description (2-3 sentences)
3. A relevant code snippet showing the bug

Format:
TITLE: <title>
DESCRIPTION: <description>
CODE:
```{language.lower()}
<code snippet>
```
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=500,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        title_match = re.search(r'TITLE:\s*(.+)', response)
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=CODE:|$)', response, re.DOTALL)
        code_match = re.search(r'```[\w]*\n(.*?)```', response, re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else f"{bug_type} in {component}"
        description = desc_match.group(1).strip() if desc_match else f"Bug found in {component}"
        code = code_match.group(1).strip() if code_match else "// Code snippet unavailable"
        
        return title, description, code
    
    def generate_bug_with_templates(self, bug_type: str, component: str, 
                                   sub_component: str, language: str) -> Tuple[str, str, str]:
        """Generate bug details using templates"""
        # Select template
        templates = self.bug_templates.get(bug_type, self.bug_templates['Logic Error'])
        template = random.choice(templates)
        
        # Fill template
        title = template.format(
            function=f"{sub_component.lower().replace(' ', '_')}_handler",
            component=component,
            condition="invalid input parameters",
            edge_case="empty array",
            symptom="data corruption",
            class=f"{component}Service",
            method="processData",
            line=random.randint(10, 500),
            time=random.randint(100, 5000),
            operation="bulk update",
            data_type="large JSON objects",
            reason="inefficient algorithm",
            param="user_id",
            data="user comments",
            endpoint=f"/api/{component.lower()}",
            attack="unauthorized access",
            thread1="worker-1",
            thread2="worker-2"
        )
        
        # Generate description
        description = f"This {bug_type.lower()} occurs in the {component} component, specifically in the {sub_component} module. "
        description += f"The issue manifests when the system processes {random.choice(['user input', 'batch data', 'concurrent requests', 'edge cases'])}. "
        description += f"This bug has been classified as {random.choice(['intermittent', 'consistent', 'environment-specific', 'data-dependent'])}."
        
        # Generate code snippet
        if language == 'Python':
            code = f"""
def {sub_component.lower().replace(' ', '_')}_handler(data):
    # BUG: {bug_type} occurs here
    if not data:  # Missing proper validation
        return None
    
    result = process_data(data)
    # Missing error handling
    return result['value']  # Potential KeyError
"""
        elif language == 'JavaScript':
            code = f"""
function handle{sub_component.replace(' ', '')}(data) {{
    // BUG: {bug_type} detected
    const result = data.map(item => {{
        return item.value * 2;  // Assumes 'value' exists
    }});
    
    return result;  // No null checks
}}
"""
        else:
            code = f"// {language} code snippet showing {bug_type}"
        
        return title, description, code.strip()
    
    def generate_file_path(self, component: str, sub_component: str, 
                          language: str) -> str:
        """Generate realistic file path"""
        lang_extensions = {
            'Python': '.py',
            'JavaScript': '.js',
            'Java': '.java',
            'Go': '.go',
            'Rust': '.rs'
        }
        
        ext = lang_extensions.get(language, '.txt')
        
        # Create hierarchical path
        base_paths = {
            'Frontend': 'src/frontend',
            'Backend': 'src/backend',
            'Database': 'src/db',
            'Infrastructure': 'infra',
            'Mobile': 'mobile'
        }
        
        base = base_paths.get(component, 'src')
        sub_folder = sub_component.lower().replace(' ', '_')
        filename = f"{random.choice(['index', 'main', 'handler', 'service', 'controller'])}{ext}"
        
        return f"{base}/{sub_folder}/{filename}"
    
    def generate_single_bug(self, bug_id: int) -> BugReport:
        """Generate a single bug report"""
        # Random selections
        bug_type = random.choice(list(self.bug_templates.keys()))
        component = random.choice(list(self.components.keys()))
        sub_component = random.choice(self.components[component])
        language = random.choice(list(self.languages.keys()))
        severity = random.choice(['Critical', 'High', 'Medium', 'Low'])
        
        # Generate bug details
        if self.use_llm and random.random() < 0.3:  # Use LLM 30% of time to save resources
            try:
                title, description, code_snippet = self.generate_bug_with_llm(
                    bug_type, component, language
                )
            except Exception as e:
                logger.debug(f"LLM generation failed: {e}, using templates")
                title, description, code_snippet = self.generate_bug_with_templates(
                    bug_type, component, sub_component, language
                )
        else:
            title, description, code_snippet = self.generate_bug_with_templates(
                bug_type, component, sub_component, language
            )
        
        # Generate metadata
        file_path = self.generate_file_path(component, sub_component, language)
        
        # Create bug report
        bug = BugReport(
            bug_id=f"BUG-{bug_id:05d}",
            title=title,
            description=description,
            code_snippet=code_snippet,
            stack_trace=self._generate_stack_trace(language, file_path) if random.random() < 0.6 else None,
            severity=severity,
            priority=self.severity_priority_map[severity],
            component=component,
            sub_component=sub_component,
            bug_type=bug_type,
            language=language,
            file_path=file_path,
            line_numbers=[random.randint(10, 200) for _ in range(random.randint(1, 3))],
            functions_affected=[f"function_{i}" for i in range(random.randint(1, 4))],
            classes_affected=[f"{component}Class{i}" for i in range(random.randint(0, 2))],
            dependencies=self._generate_dependencies(file_path, random.randint(0, 5)),
            created_date=(datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
            reporter=random.choice(self.developers),
            assignee=random.choice(self.developers) if random.random() < 0.7 else None,
            status=random.choice(['Open', 'In Progress', 'Resolved', 'Closed']),
            resolution=random.choice(['Fixed', 'Won\'t Fix', 'Duplicate', None]),
            labels=random.sample(['regression', 'blocker', 'easy-fix', 'needs-investigation', 
                                 'performance', 'security', 'ui/ux', 'data-loss'], 
                                k=random.randint(1, 4))
        )
        
        return bug
    
    def _generate_stack_trace(self, language: str, file_path: str) -> str:
        """Generate realistic stack trace"""
        if language == 'Python':
            return f"""Traceback (most recent call last):
  File "{file_path}", line {random.randint(50, 150)}, in <module>
    main()
  File "{file_path}", line {random.randint(20, 49)}, in main
    result = process_data(input_data)
  File "{file_path}", line {random.randint(10, 19)}, in process_data
    raise ValueError("Invalid data format")
ValueError: Invalid data format"""
        
        elif language == 'JavaScript':
            return f"""Error: Cannot read property 'value' of undefined
    at Object.handler ({file_path}:{random.randint(10, 100)}:15)
    at processData ({file_path}:{random.randint(101, 150)}:8)
    at async Promise.all (index 0)
    at async main ({file_path}:{random.randint(151, 200)}:5)"""
        
        else:
            return f"Error in {file_path} at line {random.randint(10, 200)}"
    
    def _generate_dependencies(self, file_path: str, count: int) -> List[str]:
        """Generate related file dependencies"""
        if count == 0:
            return []
        
        base_dir = os.path.dirname(file_path)
        deps = []
        
        for i in range(count):
            dep_file = random.choice([
                f"{base_dir}/utils.py",
                f"{base_dir}/config.py",
                f"{base_dir}/../common/helpers.py",
                f"{base_dir}/models.py",
                f"{base_dir}/constants.py"
            ])
            deps.append(dep_file)
        
        return list(set(deps))  # Remove duplicates
    
    def generate_dataset(self, num_samples: int = 25000, 
                        save_path: str = "synthetic_bug_dataset.json") -> pd.DataFrame:
        """Generate complete synthetic bug dataset"""
        logger.info(f"Generating {num_samples} synthetic bug reports...")
        
        bugs = []
        for i in tqdm(range(num_samples), desc="Generating bugs"):
            bug = self.generate_single_bug(i + 1)
            bugs.append(bug.to_dict())
        
        # Convert to DataFrame
        df = pd.DataFrame(bugs)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(bugs, f, indent=2)
        
        logger.info(f"Dataset saved to {save_path}")
        
        # Print statistics
        self._print_statistics(df)
        
        return df
    
    def _print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print(f"Total bugs: {len(df)}")
        print(f"\nSeverity distribution:")
        print(df['severity'].value_counts())
        print(f"\nComponent distribution:")
        print(df['component'].value_counts())
        print(f"\nBug type distribution:")
        print(df['bug_type'].value_counts())
        print(f"\nLanguage distribution:")
        print(df['language'].value_counts())
        print(f"\nStatus distribution:")
        print(df['status'].value_counts())


def create_topoformer_dataset(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Convert bug dataset to Topoformer format with hierarchical labels"""
    
    # Create hierarchical label system
    # Level 1: Component (5 classes)
    # Level 2: Sub-component (20 classes)
    # Level 3: Bug type (5 classes)
    # Level 4: Severity (4 classes)
    
    component_map = {comp: idx for idx, comp in enumerate(df['component'].unique())}
    subcomponent_map = {comp: idx for idx, comp in enumerate(df['sub_component'].unique())}
    bugtype_map = {bt: idx for idx, bt in enumerate(df['bug_type'].unique())}
    severity_map = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    
    topoformer_data = []
    
    for _, bug in df.iterrows():
        # Combine title, description, and code for rich context
        text = f"Title: {bug['title']}\n\n"
        text += f"Description: {bug['description']}\n\n"
        text += f"Code:\n{bug['code_snippet']}\n\n"
        text += f"File: {bug['file_path']}\n"
        if bug['stack_trace']:
            text += f"Stack Trace:\n{bug['stack_trace']}"
        
        # Create hierarchical labels
        labels = {
            'component': component_map[bug['component']],
            'sub_component': subcomponent_map[bug['sub_component']],
            'bug_type': bugtype_map[bug['bug_type']],
            'severity': severity_map[bug['severity']],
            'hierarchical_label': f"{bug['component']}/{bug['sub_component']}/{bug['bug_type']}"
        }
        
        topoformer_data.append({
            'bug_id': bug['bug_id'],
            'text': text,
            'labels': labels,
            'metadata': {
                'file_path': bug['file_path'],
                'language': bug['language'],
                'dependencies': bug['dependencies'],
                'functions_affected': bug['functions_affected'],
                'classes_affected': bug['classes_affected']
            }
        })
    
    # Split into train/val/test (70/15/15)
    n = len(topoformer_data)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    np.random.shuffle(topoformer_data)
    
    splits = {
        'train': topoformer_data[:train_size],
        'validation': topoformer_data[train_size:train_size + val_size],
        'test': topoformer_data[train_size + val_size:]
    }
    
    print(f"\nTopoformer dataset created:")
    print(f"Train: {len(splits['train'])} samples")
    print(f"Validation: {len(splits['validation'])} samples")
    print(f"Test: {len(splits['test'])} samples")
    print(f"Total unique components: {len(component_map)}")
    print(f"Total unique sub-components: {len(subcomponent_map)}")
    print(f"Total unique bug types: {len(bugtype_map)}")
    
    return splits


def main():
    """Main function to generate synthetic bug dataset"""
    
    # Initialize generator
    generator = SyntheticBugGenerator(use_llm=False)  # Set to True if you have GPU for LLM
    
    # Generate dataset
    df = generator.generate_dataset(num_samples=25000)
    
    # Convert to Topoformer format
    topoformer_splits = create_topoformer_dataset(df)
    
    # Save Topoformer format
    with open('synthetic_bug_dataset_topoformer.json', 'w') as f:
        json.dump(topoformer_splits, f, indent=2)
    
    print("\nDataset generation complete!")
    print("Files created:")
    print("- synthetic_bug_dataset.json (raw bug data)")
    print("- synthetic_bug_dataset_topoformer.json (Topoformer format)")
    
    # Sample output
    print("\nSample bug report:")
    print(json.dumps(topoformer_splits['train'][0], indent=2))


if __name__ == "__main__":
    main()
