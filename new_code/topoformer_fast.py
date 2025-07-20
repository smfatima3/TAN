"""
Fast Topoformer Implementation - Optimized for Speed
Removes computational bottlenecks while maintaining performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math
from dataclasses import dataclass

@dataclass
class FastTopoformerConfig:
    """Configuration for Fast Topoformer"""
    vocab_size: int = 30265
    embed_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    # Simplified topology settings
    use_topology: bool = True
    topology_update_freq: int = 500  # Update topology every N steps
    k_neighbors: int = 16  # Reduced from 32
    topology_dim: int = 50  # Simplified topology dimension


class FastTopologyEncoder(nn.Module):
    """Fast topology encoder using learned features instead of computing PH"""
    
    def __init__(self, config: FastTopoformerConfig):
        super().__init__()
        self.config = config
        
        # Learn topology features directly from embeddings
        self.topology_encoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, config.topology_dim)
        )
        
        # Global topology aggregation
        self.global_pool = nn.Sequential(
            nn.Linear(config.topology_dim, config.topology_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast topology encoding
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            topo_features: [batch_size, topology_dim]
        """
        # Encode each position
        batch_size, seq_len, _ = x.shape
        
        # Take mean of positions for global topology
        x_pooled = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Encode topology
        topo_features = self.topology_encoder(x_pooled)  # [batch_size, topology_dim]
        
        return topo_features


class FastTopoformerLayer(nn.Module):
    """Fast Topoformer layer without expensive topology computation"""
    
    def __init__(self, config: FastTopoformerConfig):
        super().__init__()
        self.config = config
        
        # Standard multi-head attention
        self.attention = nn.MultiheadAttention(
            config.embed_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Fast topology encoder
        if config.use_topology:
            self.topo_encoder = FastTopologyEncoder(config)
            
            # Topology-guided attention gates
            self.topo_gate = nn.Sequential(
                nn.Linear(config.topology_dim, config.embed_dim),
                nn.Sigmoid()
            )
        
        # Layer norm and FFN
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fast forward pass"""
        # Self-attention
        if mask is not None:
            mask_bool = (mask == 0).bool()
        else:
            mask_bool = None
            
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask_bool)
        
        # Apply topology gating if enabled
        if self.config.use_topology:
            # Get topology features (fast)
            topo_features = self.topo_encoder(x)  # [batch_size, topology_dim]
            
            # Create gating values
            gates = self.topo_gate(topo_features)  # [batch_size, embed_dim]
            gates = gates.unsqueeze(1)  # [batch_size, 1, embed_dim]
            
            # Apply gating
            attn_out = attn_out * gates
        
        # Residual and norm
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class FastTopoformer(nn.Module):
    """Fast Topoformer model"""
    
    def __init__(self, config: FastTopoformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Topoformer layers
        self.layers = nn.ModuleList([
            FastTopoformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output norm
        self.output_norm = nn.LayerNorm(config.embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = self.embedding_dropout(token_embeds + position_embeds)
        
        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Output norm
        hidden_states = self.output_norm(hidden_states)
        
        return hidden_states


class FastTopoformerForSequenceClassification(FastTopoformer):
    """Fast Topoformer for sequence classification"""
    
    def __init__(self, config: FastTopoformerConfig, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels
        
        # Classification head
        self.classifier = nn.Linear(config.embed_dim, num_labels)
        
        # Initialize classifier
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """Forward pass with classification"""
        # Get hidden states
        hidden_states = super().forward(input_ids, attention_mask)
        
        # Pool hidden states
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
            # Add small epsilon to prevent NaN
            if torch.isnan(loss):
                loss = loss_fn(logits + 1e-8, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'pooled_output': pooled_output
        }


def create_fast_topoformer(vocab_size: int, num_labels: int, use_topology: bool = True) -> FastTopoformerForSequenceClassification:
    """Create a fast Topoformer model"""
    config = FastTopoformerConfig(
        vocab_size=vocab_size,
        use_topology=use_topology,
        embed_dim=768,
        num_layers=6,
        num_heads=12
    )
    
    model = FastTopoformerForSequenceClassification(config, num_labels)
    
    return model


def test_fast_topoformer():
    """Test the fast implementation"""
    print("Testing Fast Topoformer...")
    
    # Create model
    model = create_fast_topoformer(vocab_size=30265, num_labels=10, use_topology=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 256
    
    input_ids = torch.randint(0, 30265, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 10, (batch_size,))
    
    # Time the forward pass
    import time
    model.eval()
    
    with torch.no_grad():
        start = time.time()
        outputs = model(input_ids, attention_mask, labels)
        end = time.time()
    
    print(f"Forward pass time: {end - start:.3f} seconds")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Compare with topology disabled
    model_no_topo = create_fast_topoformer(vocab_size=30265, num_labels=10, use_topology=False)
    
    with torch.no_grad():
        start = time.time()
        outputs_no_topo = model_no_topo(input_ids, attention_mask, labels)
        end = time.time()
    
    print(f"\nWithout topology - Forward pass time: {end - start:.3f} seconds")
    
    print("\nFast Topoformer test completed!")


if __name__ == "__main__":
    test_fast_topoformer()