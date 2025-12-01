#!/usr/bin/env python3
"""
Comprehensive LEDGAR Evaluation with Distributed Data Parallel (2 GPUs)
Models: TAN, TFIDF-SVM, Longformer, BigBird, Mamba, RetNet, YOCO (No Hyena)
GPU Memory: 2x15GB = 30GB total using DDP

Usage:
    Single GPU: python ledgar_ddp_eval.py
    Multi-GPU:  torchrun --nproc_per_node=2 ledgar_ddp_eval.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    LongformerModel, LongformerConfig,
    BigBirdModel, BigBirdConfig
)

import numpy as np
from pathlib import Path
import json
import time
import logging
import math
import gc
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

# Dataset
from datasets import load_dataset

# ====================== DDP SETUP ======================

def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0

def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is main"""
    return not dist.is_initialized() or dist.get_rank() == 0

# Setup logging (only main process)
def setup_logging():
    if is_main_process():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ledgar_ddp_eval.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger(__name__)

# ====================== CONFIGURATION ======================

@dataclass
class ExperimentConfig:
    """Experiment configuration for DDP"""
    # Data parameters
    max_train_samples: Optional[int] = None  # None = all 60K
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    max_seq_length: int = 512
    
    # Training parameters
    batch_size: int = 16  # Per GPU (total = 16*2 = 32)
    num_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Evaluation
    eval_batch_size: int = 32  # Per GPU
    
    # Efficiency measurement
    measure_runtime: bool = True
    measure_memory: bool = True
    num_timing_runs: int = 100
    
    # Statistical analysis
    num_bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # DDP settings
    num_workers: int = 4
    pin_memory: bool = True
    
    # Output
    output_dir: str = './ledgar_ddp_results'
    save_predictions: bool = True

# ====================== LEDGAR DATASET ======================

LEDGAR_CLASSES = [
    "Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", 
    "Approvals", "Arbitration", "Assignments", "Assigns", "Authority", "Authorizations", 
    "Base Salary", "Benefits", "Binding Effects", "Books", "Brokers", "Capitalization", 
    "Change In Control", "Closings", "Compliance With Laws", "Confidentiality", 
    "Consent To Jurisdiction", "Consents", "Construction", "Cooperation", "Costs", 
    "Counterparts", "Death", "Defined Terms", "Definitions", "Disability", "Disclosures", 
    "Duties", "Effective Dates", "Effectiveness", "Employment", "Enforceability", 
    "Enforcements", "Entire Agreements", "Erisa", "Existence", "Expenses", "Fees", 
    "Financial Statements", "Forfeitures", "Further Assurances", "General", "Governing Laws", 
    "Headings", "Indemnifications", "Indemnity", "Insurances", "Integration", 
    "Intellectual Property", "Interests", "Interpretations", "Jurisdictions", "Liens", 
    "Litigations", "Miscellaneous", "Modifications", "No Conflicts", "No Defaults", 
    "No Waivers", "Non-Disparagement", "Notices", "Organizations", "Participations", 
    "Payments", "Positions", "Powers", "Publicity", "Qualifications", "Records", 
    "Releases", "Remedies", "Representations", "Sales", "Sanctions", "Severability", 
    "Solvency", "Specific Performance", "Submission To Jurisdiction", "Subsidiaries", 
    "Successors", "Survival", "Tax Withholdings", "Taxes", "Terminations", "Terms", 
    "Titles", "Transactions With Affiliates", "Use Of Proceeds", "Vacations", "Venues", 
    "Vesting", "Waiver Of Jury Trials", "Waivers", "Warranties", "Withholdings"
]

class LEDGARDataset(Dataset):
    """LEDGAR dataset (100 classes)"""
    
    def __init__(self, split: str, tokenizer=None, max_length: int = 512,
                 max_samples: Optional[int] = None, for_sklearn: bool = False):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_sklearn = for_sklearn
        
        if is_main_process():
            logger.info(f"Loading LEDGAR {split} split...")
        
        dataset = load_dataset("coastalcph/lex_glue", "ledgar", split=split, trust_remote_code=True)
        
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
        
        self.data = []
        self.texts = []
        self.labels_list = []
        
        for item in tqdm(dataset, desc=f"Processing {split}", disable=not is_main_process()):
            text = item.get('text', '')
            label = item['label']
            
            if not text or len(text.strip()) == 0:
                continue
            
            self.texts.append(text)
            self.labels_list.append(label)
            
            if not self.for_sklearn and self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                self.data.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.long)
                })
        
        if is_main_process():
            logger.info(f"{split}: {len(self.texts)} samples")
    
    def __len__(self):
        return len(self.texts) if self.for_sklearn else len(self.data)
    
    def __getitem__(self, idx):
        if self.for_sklearn:
            return self.texts[idx], self.labels_list[idx]
        return self.data[idx]
    
    def get_texts_and_labels(self):
        return self.texts, self.labels_list

# ====================== TAN IMPLEMENTATION ======================

@dataclass
class TANConfig:
    """TAN Configuration"""
    vocab_size: int = 30522
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    max_seq_length: int = 512
    dropout: float = 0.1
    k_neighbors: int = 32
    use_topology: bool = True
    topology_dim: int = 128
    use_lsh: bool = True
    num_hashes: int = 8
    hash_bits: int = 64
    lsh_temperature: float = 0.1
    multi_scale_k: List[int] = field(default_factory=lambda: [8, 16, 32, 64] * 3)

class TopologicalFeatureExtractor(nn.Module):
    def __init__(self, embed_dim: int, k_neighbors: int, topology_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        self.topology_dim = topology_dim
        
        self.topology_proj = nn.Linear(embed_dim, topology_dim)
        self.topology_encoder = nn.Sequential(
            nn.Linear(topology_dim, topology_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(topology_dim * 2, topology_dim),
            nn.LayerNorm(topology_dim)
        )
        self.persistence_mlp = nn.Sequential(
            nn.Linear(topology_dim, topology_dim),
            nn.ReLU(),
            nn.Linear(topology_dim, topology_dim)
        )
    
    def compute_knn_graph(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = embeddings.shape
        embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.transpose(-2, -1))
        distances = 1 - similarity
        
        mask = torch.eye(seq_len, device=distances.device).unsqueeze(0).expand(batch_size, -1, -1)
        distances = distances.masked_fill(mask.bool(), float('inf'))
        
        k = min(self.k_neighbors, seq_len - 1)
        k = max(1, k)
        
        neighbor_distances, neighbor_indices = torch.topk(distances, k, dim=-1, largest=False)
        return neighbor_distances, neighbor_indices
    
    def extract_topological_features(self, embeddings: torch.Tensor,
                                    neighbor_distances: torch.Tensor,
                                    neighbor_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape
        k = neighbor_indices.shape[-1]
        
        topo_embeddings = self.topology_proj(embeddings)
        
        batch_indices = torch.arange(batch_size, device=embeddings.device).view(-1, 1, 1)
        batch_indices = batch_indices.expand(-1, seq_len, k)
        neighbor_features = topo_embeddings[batch_indices, neighbor_indices]
        
        weights = F.softmax(-neighbor_distances, dim=-1).unsqueeze(-1)
        weighted_neighbors = (neighbor_features * weights).sum(dim=2)
        
        combined = topo_embeddings + weighted_neighbors
        topo_features = self.topology_encoder(combined)
        persistence_features = self.persistence_mlp(topo_features)
        
        return persistence_features
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        distances, indices = self.compute_knn_graph(embeddings)
        topo_features = self.extract_topological_features(embeddings, distances, indices)
        return {'features': topo_features, 'distances': distances, 'indices': indices}

class LocalitySensitiveHashing(nn.Module):
    def __init__(self, embed_dim: int, num_hashes: int, hash_bits: int, temperature: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hashes = num_hashes
        self.hash_bits = hash_bits
        self.temperature = temperature
        
        self.register_buffer('hash_proj', torch.randn(num_hashes, embed_dim, hash_bits) * 0.02)
        self.topology_bias = nn.Parameter(torch.zeros(num_hashes, hash_bits))
    
    def hash_vectors(self, vectors: torch.Tensor, topology_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        projections = torch.einsum('bsd,hdk->bshk', vectors, self.hash_proj)
        
        if topology_features is not None:
            topo_dim = topology_features.size(-1)
            if topo_dim <= self.embed_dim:
                topo_proj_matrix = self.hash_proj[:, :topo_dim, :]
                topo_proj = torch.einsum('bsd,hdk->bshk', topology_features, topo_proj_matrix)
                projections = projections + self.temperature * topo_proj
        
        projections = projections + self.topology_bias.unsqueeze(0).unsqueeze(1)
        hash_codes = torch.sign(projections)
        return hash_codes
    
    def compute_hash_similarity(self, query_hashes: torch.Tensor, key_hashes: torch.Tensor) -> torch.Tensor:
        similarity = torch.einsum('bqhk,bkhk->bhqk', query_hashes, key_hashes) / self.hash_bits
        similarity = similarity.mean(dim=1)
        return similarity
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                topology_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_hashes = self.hash_vectors(queries, topology_features)
        key_hashes = self.hash_vectors(keys, topology_features)
        hash_similarity = self.compute_hash_similarity(query_hashes, key_hashes)
        attention_mask = (hash_similarity > 0.3).float()
        return attention_mask

class TopologicalAttention(nn.Module):
    def __init__(self, config: TANConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.use_topology = config.use_topology
        self.use_lsh = config.use_lsh
        
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        if self.use_topology:
            self.topology_gate = nn.Sequential(
                nn.Linear(config.embed_dim + config.topology_dim, config.embed_dim),
                nn.Sigmoid()
            )
        
        if self.use_lsh:
            self.lsh = LocalitySensitiveHashing(
                config.embed_dim, config.num_hashes, config.hash_bits, config.lsh_temperature
            )
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, hidden_states: torch.Tensor,
                topology_features: Optional[Dict[str, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        
        if self.use_lsh and topology_features is not None:
            try:
                lsh_mask = self.lsh(hidden_states, hidden_states, topology_features.get('features'))
                lsh_mask = lsh_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                attention_scores = attention_scores.masked_fill(lsh_mask == 0, -1e4)
            except:
                pass
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            mask_expanded = mask_expanded.expand(batch_size, self.num_heads, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e4)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        if self.use_topology and topology_features is not None:
            try:
                topo_features = topology_features['features']
                combined = torch.cat([context, topo_features], dim=-1)
                gate = self.topology_gate(combined)
                context = gate * context + (1 - gate) * hidden_states
            except:
                pass
        
        output = self.out_proj(context)
        output = self.dropout(output)
        return output

class TANLayer(nn.Module):
    def __init__(self, config: TANConfig):
        super().__init__()
        self.config = config
        
        if config.use_topology:
            self.topology_extractor = TopologicalFeatureExtractor(
                config.embed_dim, config.k_neighbors, config.topology_dim
            )
        
        self.attention = TopologicalAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        topology_features = None
        if self.config.use_topology:
            try:
                topology_features = self.topology_extractor(hidden_states)
            except:
                pass
        
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, topology_features, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class TAN(nn.Module):
    def __init__(self, config: TANConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding_norm = nn.LayerNorm(config.embed_dim)
        
        self.layers = nn.ModuleList([TANLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.final_norm(hidden_states)
        
        return {'hidden_states': hidden_states}

class TANForLegalClassification(nn.Module):
    def __init__(self, config: TANConfig, num_labels: int = 100):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.tan = TAN(config)
        
        self.pooler = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, num_labels)
        )
        
        self.name = "TAN"
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.tan(input_ids, attention_mask)
        hidden_states = outputs['hidden_states']
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states[:, 0, :]
        
        pooled_output = self.pooler(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ====================== BASELINE MODELS ======================

class TFIDF_SVM:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.classifier = LinearSVC(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
        self.name = "TFIDF-SVM"
    
    def train(self, train_dataset):
        texts, labels = train_dataset.get_texts_and_labels()
        X_train = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X_train, labels)
        return {'trained': True}
    
    def evaluate(self, eval_dataset):
        texts, labels = eval_dataset.get_texts_and_labels()
        X_eval = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X_eval)
        
        metrics = {
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_micro': f1_score(labels, predictions, average='micro'),
            'accuracy': accuracy_score(labels, predictions)
        }
        return metrics, predictions
    
    def get_efficiency_metrics(self):
        n_features = len(self.vectorizer.get_feature_names_out())
        params = n_features * 100
        return {'total_params': params, 'total_params_M': params / 1e6}

class LongformerForLEDGAR(nn.Module):
    def __init__(self, num_labels=100, max_length=512):
        super().__init__()
        config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        config.num_labels = num_labels
        config.attention_window = [128] * config.num_hidden_layers
        
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.name = "Longformer"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

class BigBirdForLEDGAR(nn.Module):
    def __init__(self, num_labels=100, max_length=512):
        super().__init__()
        config = BigBirdConfig.from_pretrained('google/bigbird-roberta-base')
        config.num_labels = num_labels
        config.attention_type = "block_sparse"
        config.block_size = 64
        config.num_random_blocks = 3
        
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base', config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.name = "BigBird"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bigbird(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

class MambaForLEDGAR(nn.Module):
    def __init__(self, num_labels=100, d_model=768, n_layer=12):
        super().__init__()
        
        try:
            from mamba_ssm import Mamba
            
            self.embedding = nn.Embedding(30522, d_model)
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layer)
            ])
            
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, num_labels)
            self.name = "Mamba"
            self.available = True
            
        except ImportError:
            if is_main_process():
                logger.warning("Mamba not available")
            self.available = False
            self.name = "Mamba (unavailable)"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        if not self.available:
            raise ImportError("Mamba not installed")
        
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x) + x
        
        x = self.norm(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(1) / mask.sum(1)
        else:
            x = x.mean(1)
        
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

class RetNetForLEDGAR(nn.Module):
    def __init__(self, num_labels=100, d_model=768, n_layer=12):
        super().__init__()
        
        try:
            from torchscale.architecture.retnet import RetNetDecoder
            from torchscale.architecture.config import RetNetConfig
            
            config = RetNetConfig(
                decoder_embed_dim=d_model,
                decoder_layers=n_layer,
                decoder_retention_heads=12,
                decoder_ffn_embed_dim=d_model * 4
            )
            
            self.retnet = RetNetDecoder(config, embed_tokens=None)
            self.embedding = nn.Embedding(30522, d_model)
            self.classifier = nn.Linear(d_model, num_labels)
            self.name = "RetNet"
            self.available = True
            
        except ImportError:
            if is_main_process():
                logger.warning("RetNet not available")
            self.available = False
            self.name = "RetNet (unavailable)"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        if not self.available:
            raise ImportError("RetNet not installed")
        
        x = self.embedding(input_ids)
        outputs = self.retnet(x, features_only=True)
        pooled = outputs[0].mean(1)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

class YOCOForLEDGAR(nn.Module):
    def __init__(self, num_labels=100, d_model=768, n_layer=12):
        super().__init__()
        
        self.embedding = nn.Embedding(30522, d_model)
        
        self.self_decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=12, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layer // 2)
        ])
        
        self.cross_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=12, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layer // 2)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_labels)
        self.name = "YOCO"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        
        for layer in self.self_decoder:
            x = layer(x)
        
        kv_cache = x
        
        for layer in self.cross_decoder:
            x = layer(x, kv_cache)
        
        x = self.norm(x)
        x = x.mean(1)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ====================== EFFICIENCY & STATISTICS ======================

class EfficiencyMetrics:
    def __init__(self, device='cuda'):
        self.device = device
    
    def count_parameters(self, model):
        # Handle DDP wrapped models
        if isinstance(model, DDP):
            model = model.module
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_params': total,
            'trainable_params': trainable,
            'total_params_M': total / 1e6
        }
    
    def measure_inference_time(self, model, dataloader, num_runs=100):
        # Unwrap DDP
        if isinstance(model, DDP):
            model = model.module
        
        model.eval()
        times = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    _ = model(**{k: v.to(self.device) for k, v in batch.items() if k != 'labels'})
                break
        
        with torch.no_grad():
            for _ in range(num_runs):
                for batch in dataloader:
                    start = time.time()
                    
                    if isinstance(batch, dict):
                        _ = model(**{k: v.to(self.device) for k, v in batch.items() if k != 'labels'})
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    times.append(time.time() - start)
                    break
        
        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000
        }
    
    def measure_memory(self, model, dataloader):
        if self.device != 'cuda':
            return {'peak_memory_mb': 0}
        
        # Unwrap DDP
        if isinstance(model, DDP):
            model = model.module
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    _ = model(**{k: v.to(self.device) for k, v in batch.items() if k != 'labels'})
                break
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {
            'peak_memory_mb': peak_memory,
            'peak_memory_gb': peak_memory / 1024
        }

class StatisticalAnalysis:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def bootstrap_confidence_interval(self, y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
        scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score = metric_fn(np.array(y_true)[indices], np.array(y_pred)[indices])
            scores.append(score)
        
        alpha = 1 - confidence
        lower = np.percentile(scores, alpha/2 * 100)
        upper = np.percentile(scores, (1 - alpha/2) * 100)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_lower': lower,
            'ci_upper': upper
        }

# ====================== DDP TRAINER ======================

class DDPModelTrainer:
    def __init__(self, model, config: ExperimentConfig, model_name: str, rank: int, local_rank: int):
        self.config = config
        self.model_name = model_name
        self.rank = rank
        self.local_rank = local_rank
        self.device = f'cuda:{local_rank}'
        
        # Move model to device and wrap with DDP
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    def train(self, train_loader, val_loader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        training_stats = []
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loader.sampler.set_epoch(epoch)  # Important for DDP
            
            total_loss = 0
            
            if is_main_process():
                pbar = tqdm(train_loader, desc=f"{self.model_name} Epoch {epoch+1}/{self.config.num_epochs}")
            else:
                pbar = train_loader
            
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if is_main_process():
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation (only on main process for simplicity)
            if is_main_process():
                val_metrics, _ = self.evaluate(val_loader)
                logger.info(f"{self.model_name} Epoch {epoch+1}: Loss={avg_loss:.4f}, F1={val_metrics['f1_macro']:.4f}")
                
                training_stats.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    **val_metrics
                })
                
                if val_metrics['f1_macro'] > best_f1:
                    best_f1 = val_metrics['f1_macro']
            
            # Sync across processes
            if dist.is_initialized():
                dist.barrier()
        
        return training_stats
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                outputs = self.model(**batch)
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_micro': f1_score(all_labels, all_preds, average='micro'),
            'accuracy': accuracy_score(all_labels, all_preds)
        }
        
        return metrics, all_preds

# ====================== MAIN EXPERIMENT ======================

class ComprehensiveDDPExperiment:
    def __init__(self, config: ExperimentConfig, rank: int, world_size: int, local_rank: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = f'cuda:{local_rank}'
        
        self.results = {}
        self.predictions = {}
        self.efficiency_metrics = EfficiencyMetrics(self.device)
        self.statistical_analysis = StatisticalAnalysis(config)
        
        if is_main_process():
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_datasets(self):
        if is_main_process():
            logger.info("Loading datasets...")
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.train_dataset = LEDGARDataset('train', tokenizer, self.config.max_seq_length, self.config.max_train_samples)
        self.val_dataset = LEDGARDataset('validation', tokenizer, self.config.max_seq_length, self.config.max_val_samples)
        self.test_dataset = LEDGARDataset('test', tokenizer, self.config.max_seq_length, self.config.max_test_samples)
        
        # For sklearn (only on main process)
        if is_main_process():
            self.train_dataset_sklearn = LEDGARDataset('train', None, self.config.max_seq_length, self.config.max_train_samples, for_sklearn=True)
            self.test_dataset_sklearn = LEDGARDataset('test', None, self.config.max_seq_length, self.config.max_test_samples, for_sklearn=True)
    
    def create_ddp_dataloaders(self, dataset, batch_size, shuffle=False):
        """Create DDP-compatible dataloader"""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return loader
    
    def run_tfidf_svm(self):
        """TFIDF-SVM only on main process"""
        if not is_main_process():
            return
        
        logger.info("\n" + "="*60)
        logger.info("Running TFIDF-SVM")
        logger.info("="*60)
        
        model = TFIDF_SVM(self.config)
        
        start_time = time.time()
        model.train(self.train_dataset_sklearn)
        train_time = time.time() - start_time
        
        test_metrics, predictions = model.evaluate(self.test_dataset_sklearn)
        
        efficiency = model.get_efficiency_metrics()
        efficiency['train_time_sec'] = train_time
        
        self.results['TFIDF-SVM'] = {
            'performance': test_metrics,
            'efficiency': efficiency
        }
        self.predictions['TFIDF-SVM'] = predictions
        
        logger.info(f"TFIDF-SVM: F1-Macro={test_metrics['f1_macro']:.4f}")
    
    def run_neural_model(self, model_class, model_name, model_kwargs=None):
        if is_main_process():
            logger.info("\n" + "="*60)
            logger.info(f"Running {model_name}")
            logger.info("="*60)
        
        try:
            if model_kwargs is None:
                model_kwargs = {}
            model = model_class(num_labels=100, **model_kwargs)
            
            if hasattr(model, 'available') and not model.available:
                if is_main_process():
                    logger.warning(f"{model_name} not available, skipping")
                return
            
            # Create DDP dataloaders
            train_loader = self.create_ddp_dataloaders(self.train_dataset, self.config.batch_size, shuffle=True)
            val_loader = self.create_ddp_dataloaders(self.val_dataset, self.config.eval_batch_size, shuffle=False)
            test_loader = self.create_ddp_dataloaders(self.test_dataset, self.config.eval_batch_size, shuffle=False)
            
            # Train with DDP
            trainer = DDPModelTrainer(model, self.config, model_name, self.rank, self.local_rank)
            
            start_time = time.time()
            training_stats = trainer.train(train_loader, val_loader)
            train_time = time.time() - start_time
            
            # Evaluation only on main process
            if is_main_process():
                test_metrics, predictions = trainer.evaluate(test_loader)
                
                efficiency = {}
                efficiency.update(self.efficiency_metrics.count_parameters(trainer.model))
                efficiency['train_time_sec'] = train_time
                
                if self.config.measure_runtime:
                    timing = self.efficiency_metrics.measure_inference_time(
                        trainer.model, test_loader, self.config.num_timing_runs
                    )
                    efficiency.update(timing)
                
                if self.config.measure_memory:
                    memory = self.efficiency_metrics.measure_memory(trainer.model, test_loader)
                    efficiency.update(memory)
                
                self.results[model_name] = {
                    'performance': test_metrics,
                    'efficiency': efficiency,
                    'training_stats': training_stats
                }
                self.predictions[model_name] = predictions
                
                logger.info(f"{model_name}: F1-Macro={test_metrics['f1_macro']:.4f}, Params={efficiency['total_params_M']:.1f}M")
            
            # Cleanup
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            # Sync across processes
            if dist.is_initialized():
                dist.barrier()
            
        except Exception as e:
            if is_main_process():
                logger.error(f"Error running {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def run_all_models(self):
        # Classical (main process only)
        self.run_tfidf_svm()
        
        # Sync before neural models
        if dist.is_initialized():
            dist.barrier()
        
        # Sparse attention
        self.run_neural_model(LongformerForLEDGAR, "Longformer")
        self.run_neural_model(BigBirdForLEDGAR, "BigBird")
        
        # Recent models
        self.run_neural_model(MambaForLEDGAR, "Mamba", {'n_layer': 12})
        self.run_neural_model(RetNetForLEDGAR, "RetNet", {'n_layer': 12})
        self.run_neural_model(YOCOForLEDGAR, "YOCO", {'n_layer': 12})
        
        # TAN
        tan_config = TANConfig(
            vocab_size=30522,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_length=512,
            k_neighbors=32,
            use_topology=True,
            use_lsh=True
        )
        self.run_neural_model(lambda **kwargs: TANForLegalClassification(tan_config, **kwargs), "TAN")
    
    def perform_statistical_analysis(self):
        if not is_main_process():
            return
        
        logger.info("\n" + "="*60)
        logger.info("Statistical Analysis")
        logger.info("="*60)
        
        _, true_labels = self.test_dataset_sklearn.get_texts_and_labels()
        
        statistical_results = {}
        
        for model_name, preds in self.predictions.items():
            f1_macro_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
            
            ci = self.statistical_analysis.bootstrap_confidence_interval(
                true_labels, preds, f1_macro_fn,
                n_bootstrap=self.config.num_bootstrap_samples,
                confidence=self.config.confidence_level
            )
            
            statistical_results[model_name] = {'bootstrap_ci': ci}
            logger.info(f"{model_name}: F1={ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        
        self.results['statistical_analysis'] = statistical_results
    
    def save_results(self):
        if not is_main_process():
            return
        
        output_file = Path(self.config.output_dir) / 'comprehensive_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        self.create_summary_table()
    
    def create_summary_table(self):
        if not is_main_process():
            return
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE LEDGAR EVALUATION RESULTS (DDP)")
        logger.info("="*80)
        
        logger.info("\nPERFORMANCE METRICS:")
        logger.info("-" * 80)
        logger.info(f"{'Model':<20} {'F1-Macro':>12} {'F1-Micro':>12} {'Accuracy':>12}")
        logger.info("-" * 80)
        
        for model_name, results in sorted(self.results.items()):
            if model_name == 'statistical_analysis':
                continue
            perf = results['performance']
            logger.info(f"{model_name:<20} {perf['f1_macro']:>12.4f} {perf['f1_micro']:>12.4f} {perf['accuracy']:>12.4f}")
        
        logger.info("\nEFFICIENCY METRICS:")
        logger.info("-" * 80)
        logger.info(f"{'Model':<20} {'Params (M)':>12} {'Infer (ms)':>12} {'Memory (MB)':>12}")
        logger.info("-" * 80)
        
        for model_name, results in sorted(self.results.items()):
            if model_name == 'statistical_analysis':
                continue
            eff = results['efficiency']
            params = eff.get('total_params_M', 0)
            infer = eff.get('mean_inference_time_ms', 0)
            memory = eff.get('peak_memory_mb', 0)
            
            logger.info(f"{model_name:<20} {params:>12.2f} {infer:>12.2f} {memory:>12.1f}")

# ====================== MAIN ======================

def main():
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Setup logging
    global logger
    logger = setup_logging()
    
    if is_main_process():
        logger.info(f"Running on {world_size} GPU(s)")
        logger.info(f"Each GPU: ~15GB, Total: {world_size * 15}GB")
    
    # Configuration
    config = ExperimentConfig(
        max_train_samples=None,  # Use full 60K
        num_epochs=20,
        batch_size=16,  # Per GPU
        measure_runtime=True,
        measure_memory=True,
        output_dir='./ledgar_ddp_results'
    )
    
    try:
        # Run experiment
        experiment = ComprehensiveDDPExperiment(config, rank, world_size, local_rank)
        experiment.load_datasets()
        experiment.run_all_models()
        experiment.perform_statistical_analysis()
        experiment.save_results()
        
        if is_main_process():
            logger.info("\n✓ Comprehensive LEDGAR DDP evaluation complete!")
    
    finally:
        # Cleanup
        cleanup_ddp()

if __name__ == "__main__":
    main()
