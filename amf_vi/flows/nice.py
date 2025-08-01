import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow

class CouplingLayer(nn.Module):
    """NICE coupling layer implementation."""
    
    def __init__(self, dim: int, hidden_dim: int = 64, mask_type: str = 'alternating'):
        super().__init__()
        self.dim = dim
        
        # Create mask - alternating pattern
        if mask_type == 'alternating':
            self.register_buffer('mask', torch.arange(dim) % 2)
        else:
            # Half-half split
            mask = torch.zeros(dim)
            mask[:dim//2] = 1
            self.register_buffer('mask', mask)
        
        # Coupling function m(x) - can be arbitrarily complex
        input_dim = int(self.mask.sum().item())
        output_dim = dim - input_dim
        
        self.coupling_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward transformation: y1 = x1, y2 = x2 + m(x1)"""
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        # Get input for coupling function (masked part)
        coupling_input = x_masked[self.mask.bool()].view(x.size(0), -1)
        
        if coupling_input.size(1) > 0:
            coupling_output = self.coupling_net(coupling_input)
            
            # Apply additive coupling
            y = x_masked.clone()
            y[:, ~self.mask.bool()] = x_unmasked[~self.mask.bool()].view(x.size(0), -1) + coupling_output
        else:
            y = x
        
        return y
    
    def inverse(self, y):
        """Inverse transformation: x1 = y1, x2 = y2 - m(y1)"""
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)
        
        # Get input for coupling function (masked part)
        coupling_input = y_masked[self.mask.bool()].view(y.size(0), -1)
        
        if coupling_input.size(1) > 0:
            coupling_output = self.coupling_net(coupling_input)
            
            # Apply inverse additive coupling
            x = y_masked.clone()
            x[:, ~self.mask.bool()] = y_unmasked[~self.mask.bool()].view(y.size(0), -1) - coupling_output
        else:
            x = y
        
        return x

class NICEFlow(BaseFlow):
    """NICE: Non-linear Independent Components Estimation."""
    
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Create alternating coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(n_layers):
            # Alternate mask pattern
            mask_type = 'alternating' if i % 2 == 0 else 'alternating'
            layer = CouplingLayer(dim, hidden_dim, mask_type)
            # Flip mask for every other layer
            if i % 2 == 1:
                layer.mask = 1 - layer.mask
            self.coupling_layers.append(layer)
        
        # Diagonal scaling layer (learnable)
        self.log_scale = nn.Parameter(torch.zeros(dim))
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through NICE layers."""
        z = x
        
        # Apply coupling layers (Jacobian determinant is 1 for each)
        for layer in self.coupling_layers:
            z = layer.forward(z)
        
        # Apply diagonal scaling
        z = z * torch.exp(self.log_scale)
        
        # Log determinant is sum of log scales
        log_det = self.log_scale.sum().expand(x.size(0))
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        # Inverse scaling
        x = z * torch.exp(-self.log_scale)
        
        # Apply inverse coupling layers in reverse order
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)
        
        return x