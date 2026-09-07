import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder layer for capturing global contextual spatial relationships."""
    def __init__(self, embed_dim, num_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # Self Attention + Residual
        src2 = self.norm1(src)
        attn_out, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(attn_out)
        
        # Feed Forward + Residual
        src2 = self.norm2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff_out)
        return src

class MobileNetViTHybrid(nn.Module):
    """Hybrid Network fusing MobileNetV3 local representations with Transformer global attention."""
    def __init__(self, num_classes=38, embed_dim=160, num_heads=8, num_transformer_layers=2):
        super().__init__()
        
        # MobileNetV3 Convolutional Backbone
        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)
        self.feature_extractor = backbone.features  # Outputs (B, 960, 7, 7) for 224x224 input

        # Bottleneck projection down to embed_dim (160)
        self.projection = nn.Conv2d(960, embed_dim, kernel_size=1)
        
        # Spatial Positional Embeddings for the 7x7=49 spatial patches
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim))
        
        # Vision Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])
        
        # Classification Head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Local CNN Feature Extraction: (B, 3, 224, 224) -> (B, 960, 7, 7)
        features = self.feature_extractor(x)
        
        # Dimension Reduction: (B, 960, 7, 7) -> (B, 160, 7, 7)
        proj_features = self.projection(features)
        
        # Flatten spatial grid to tokens: (B, 160, 7, 7) -> (B, 160, 49) -> (B, 49, 160)
        batch_size, channels, h, w = proj_features.shape
        tokens = proj_features.flatten(2).permute(0, 2, 1)
        
        # Add positional embedding
        tokens = tokens + self.pos_embedding
        
        # Global Transformer Context Reasoning
        for transformer in self.transformer_layers:
            tokens = transformer(tokens)
            
        tokens = self.layer_norm(tokens)
        
        # Global Average Pooling across spatial patch tokens
        global_repr = tokens.mean(dim=1)
        
        # Multi-crop classification output logits
        logits = self.classifier(global_repr)
        return logits