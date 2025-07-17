"""
Baseline Models for Topoformer Comparison
Includes: BERT, CodeBERT, Longformer, and FlashAttention-2 implementations
"""

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertModel,
    RobertaForSequenceClassification,
    RobertaModel,
    LongformerForSequenceClassification,
    LongformerModel,
    AutoConfig
)
from typing import Dict, Optional, Tuple
import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import FlashAttention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.warning("FlashAttention not available. Install with: pip install flash-attn")


class BaselineModel:
    """Base class for all baseline models"""
    
    def __init__(self, model_name: str, num_labels: int, device: str = 'cuda'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**3
        return 0.0
    
    def benchmark_forward_pass(self, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor,
                             num_runs: int = 10) -> Dict[str, float]:
        """Benchmark forward pass performance"""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.time() - start_time
        
        return {
            'avg_time_ms': (total_time / num_runs) * 1000,
            'throughput': num_runs / total_time,
            'memory_gb': self.get_memory_usage()
        }


class BERTBaseline(BaselineModel):
    """BERT-large baseline model"""
    
    def __init__(self, num_labels: int, device: str = 'cuda'):
        super().__init__('google-bert/bert-large-cased', num_labels, device)
        
        logger.info(f"Loading BERT-large baseline...")
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        logger.info(f"BERT model loaded with {self.count_parameters():,} parameters")


class CodeBERTBaseline(BaselineModel):
    """CodeBERT baseline for code understanding tasks"""
    
    def __init__(self, num_labels: int, device: str = 'cuda'):
        super().__init__('microsoft/codebert-base', num_labels, device)
        
        logger.info(f"Loading CodeBERT baseline...")
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        logger.info(f"CodeBERT model loaded with {self.count_parameters():,} parameters")


class LongformerBaseline(BaselineModel):
    """Longformer baseline for long document understanding"""
    
    def __init__(self, num_labels: int, max_length: int = 4096, device: str = 'cuda'):
        super().__init__('allenai/longformer-base-4096', num_labels, device)
        self.max_length = max_length
        
        logger.info(f"Loading Longformer baseline...")
        
        # Configure for classification
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = num_labels
        
        self.model = LongformerForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        logger.info(f"Longformer model loaded with {self.count_parameters():,} parameters")
    
    def prepare_global_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create global attention mask for CLS token"""
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1  # CLS token gets global attention
        return global_attention_mask


class FlashAttentionBERT(nn.Module):
    """BERT model with FlashAttention-2 implementation"""
    
    def __init__(self, num_labels: int, model_name: str = 'bert-base-uncased'):
        super().__init__()
        
        if not FLASH_ATTENTION_AVAILABLE:
            raise ImportError("FlashAttention not available. Please install flash-attn package.")
        
        # Load base BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Replace attention with FlashAttention
        self._replace_attention_with_flash()
        
        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
    def _replace_attention_with_flash(self):
        """Replace BERT attention layers with FlashAttention"""
        for layer in self.bert.encoder.layer:
            # Store original attention
            original_attention = layer.attention.self
            
            # Create FlashAttention wrapper
            layer.attention.self = FlashAttentionWrapper(
                self.config.hidden_size,
                self.config.num_attention_heads,
                self.config.attention_probs_dropout_prob
            )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with FlashAttention"""
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool and classify
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


class FlashAttentionWrapper(nn.Module):
    """Wrapper to use FlashAttention in place of standard attention"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using FlashAttention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply FlashAttention
        if FLASH_ATTENTION_AVAILABLE:
            # FlashAttention expects (batch, seqlen, nheads, headdim)
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False
            )
        else:
            # Fallback to standard attention
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape output
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return attn_output
    
    def _standard_attention(self, q, k, v, attention_mask):
        """Standard attention as fallback"""
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        # Apply softmax and dropout
        attn_probs = nn.functional.softmax(scores, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Transpose back
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output


class StandardBERTAttention(BaselineModel):
    """Standard BERT with explicit attention computation for comparison"""
    
    def __init__(self, num_labels: int, device: str = 'cuda'):
        super().__init__('bert-base-uncased', num_labels, device)
        
        logger.info("Loading standard BERT with explicit attention...")
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=True  # Enable attention output
        ).to(self.device)
        
        logger.info(f"Standard BERT loaded with {self.count_parameters():,} parameters")
    
    def get_attention_weights(self, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights for analysis"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Returns attention weights for each layer
        return outputs.attentions


def create_baseline_model(model_type: str, num_labels: int, 
                         device: str = 'cuda') -> BaselineModel:
    """Factory function to create baseline models"""
    
    model_map = {
        'bert': BERTBaseline,
        'codebert': CodeBERTBaseline,
        'longformer': LongformerBaseline,
        'bert_standard': StandardBERTAttention
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_map.keys())}")
    
    return model_map[model_type](num_labels, device)


def benchmark_all_baselines(num_labels: int = 5, 
                          batch_size: int = 4,
                          seq_length: int = 128,
                          device: str = 'cuda'):
    """Benchmark all baseline models"""
    print("Benchmarking Baseline Models")
    print("=" * 50)
    
    # Create dummy input
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device_obj)
    attention_mask = torch.ones(batch_size, seq_length).to(device_obj)
    
    results = {}
    
    # Test each baseline
    baselines = ['bert', 'codebert', 'longformer', 'bert_standard']
    
    for baseline_name in baselines:
        print(f"\nTesting {baseline_name}...")
        try:
            # Create model
            model = create_baseline_model(baseline_name, num_labels, device)
            
            # Benchmark
            bench_results = model.benchmark_forward_pass(input_ids, attention_mask, num_runs=5)
            
            results[baseline_name] = {
                'parameters': model.count_parameters(),
                'avg_time_ms': bench_results['avg_time_ms'],
                'throughput': bench_results['throughput'],
                'memory_gb': bench_results['memory_gb']
            }
            
            print(f"✓ {baseline_name}: {model.count_parameters():,} params, "
                  f"{bench_results['avg_time_ms']:.2f}ms/batch, "
                  f"{bench_results['memory_gb']:.2f}GB memory")
            
        except Exception as e:
            print(f"✗ {baseline_name} failed: {e}")
            results[baseline_name] = {'error': str(e)}
    
    # Test FlashAttention if available
    if FLASH_ATTENTION_AVAILABLE:
        print("\nTesting FlashAttention BERT...")
        try:
            flash_model = FlashAttentionBERT(num_labels).to(device_obj)
            
            # Warmup and benchmark
            with torch.no_grad():
                _ = flash_model(input_ids, attention_mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(5):
                with torch.no_grad():
                    _ = flash_model(input_ids, attention_mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.time() - start_time
            
            results['flash_attention'] = {
                'parameters': sum(p.numel() for p in flash_model.parameters() if p.requires_grad),
                'avg_time_ms': (total_time / 5) * 1000,
                'throughput': 5 / total_time,
                'memory_gb': torch.cuda.memory_allocated(device_obj) / 1024**3
            }
            
            print(f"✓ FlashAttention: {results['flash_attention']['avg_time_ms']:.2f}ms/batch")
            
        except Exception as e:
            print(f"✗ FlashAttention failed: {e}")
    
    return results


def compare_attention_patterns(topoformer_attn: torch.Tensor,
                             baseline_attn: torch.Tensor) -> Dict[str, float]:
    """Compare attention patterns between models"""
    
    # Compute various similarity metrics
    # Ensure same shape
    if topoformer_attn.shape != baseline_attn.shape:
        # Interpolate or pad to match shapes
        min_size = min(topoformer_attn.shape[-1], baseline_attn.shape[-1])
        topoformer_attn = topoformer_attn[..., :min_size, :min_size]
        baseline_attn = baseline_attn[..., :min_size, :min_size]
    
    # Flatten attention matrices
    topo_flat = topoformer_attn.flatten()
    base_flat = baseline_attn.flatten()
    
    # Compute metrics
    metrics = {
        'cosine_similarity': torch.nn.functional.cosine_similarity(
            topo_flat.unsqueeze(0), base_flat.unsqueeze(0)
        ).item(),
        'l2_distance': torch.norm(topo_flat - base_flat).item(),
        'kl_divergence': torch.nn.functional.kl_div(
            torch.log_softmax(topo_flat, dim=0),
            torch.softmax(base_flat, dim=0),
            reduction='sum'
        ).item()
    }
    
    return metrics


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_all_baselines(
        num_labels=5,
        batch_size=4,
        seq_length=128
    )
    
    print("\n" + "="*50)
    print("Benchmark Summary:")
    print("="*50)
    
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"\n{model_name}:")
            print(f"  Parameters: {metrics['parameters']:,}")
            print(f"  Avg Time: {metrics['avg_time_ms']:.2f}ms")
            print(f"  Memory: {metrics['memory_gb']:.2f}GB")