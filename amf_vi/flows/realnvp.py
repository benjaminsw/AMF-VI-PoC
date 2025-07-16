import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow

class SimpleMLP(nn.Module):
    """Simple MLP for Real-NVP."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RealNVPFlow(BaseFlow):
    """Real-NVP coupling flow for PoC."""
    
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__(dim)
        
        # Create alternating masks
        self.masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1  # Alternate between even/odd indices
            self.register_buffer(f'mask_{i}', mask)
            self.masks.append(mask)
        
        # Create scale and translation networks
        self.scale_nets = nn.ModuleList()
        self.translate_nets = nn.ModuleList()
        
        for _ in range(n_layers):
            self.scale_nets.append(SimpleMLP(dim, dim, hidden_dim))
            self.translate_nets.append(SimpleMLP(dim, dim, hidden_dim))
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i, mask in enumerate(self.masks):
            z_masked = z * mask
            
            # Compute scale and translate
            s = self.scale_nets[i](z_masked) * (1 - mask)
            t = self.translate_nets[i](z_masked) * (1 - mask)
            
            # Apply transformation
            z = z_masked + (1 - mask) * (z * torch.exp(s) + t)
            
            # Accumulate log determinant
            log_det_total += s.sum(dim=1)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(len(self.masks))):
            mask = self.masks[i]
            x_masked = x * mask
            
            s = self.scale_nets[i](x_masked) * (1 - mask)
            t = self.translate_nets[i](x_masked) * (1 - mask)
            
            x = x_masked + (1 - mask) * ((x - t) * torch.exp(-s))
        
        return x