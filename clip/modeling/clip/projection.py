import torch
from torch import nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

def build_projection(cfg):
    proj_dim = cfg.MODEL.CLIP.PROJECTION.DIM
    vid_dim = cfg.MODEL.CLIP.VIDEOEMBED.DIM
    text_dim = cfg.MODEL.CLIP.TEXTEMBED.DIM
    return ProjectionHead(vid_dim, proj_dim), ProjectionHead(text_dim, proj_dim)