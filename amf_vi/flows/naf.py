import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base_flow import BaseFlow

class MaskedLinear(nn.Module):
    """Masked linear layer for autoregressive neural networks."""
    
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Ensure mask has correct shape
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(out_features, -1)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)

class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation for NAF."""
    
    def __init__(self, input_dim, hidden_dim=64, output_multiplier=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim * output_multiplier
        
        # Create autoregressive masks - each dimension can only depend on previous ones
        # Input to hidden mask
        input_mask = torch.zeros(hidden_dim, input_dim)
        degrees = torch.randint(0, input_dim - 1, (hidden_dim,))
        for h in range(hidden_dim):
            input_mask[h, :degrees[h] + 1] = 1
        
        # Hidden to output mask
        output_mask = torch.zeros(self.output_dim, hidden_dim)
        for d in range(input_dim):
            for param_idx in range(output_multiplier):
                out_idx = d * output_multiplier + param_idx
                if out_idx < self.output_dim:
                    for h in range(hidden_dim):
                        if degrees[h] < d:
                            output_mask[out_idx, h] = 1
        
        self.input_layer = MaskedLinear(input_dim, hidden_dim, input_mask)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = MaskedLinear(hidden_dim, self.output_dim, output_mask)
        
        self.activation = nn.ReLU()
        self.register_buffer('degrees', degrees)
    
    def forward(self, x):
        h = self.activation(self.input_layer(x))
        h = self.activation(self.hidden_layer(h))
        return self.output_layer(h)

class NAFFlowSimplified(BaseFlow):
    """Simplified Neural Autoregressive Flow for better performance and stability."""
    
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 32):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Use neural network conditioned affine transformations
        self.conditioners = nn.ModuleList()
        
        for i in range(n_layers):
            # Each conditioner outputs scale and shift parameters
            conditioner = MADE(dim, hidden_dim, output_multiplier=2)  # 2 for scale and shift
            self.conditioners.append(conditioner)
        
        # Permutations for increased expressiveness
        self.permutations = []
        for i in range(n_layers):
            if i == 0:
                perm = torch.arange(dim)
            else:
                perm = torch.randperm(dim)
            self.register_buffer(f'perm_{i}', perm)
            self.permutations.append(perm)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation with neural conditioning."""
        z = x.clone()
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for layer_idx in range(self.n_layers):
            # Apply permutation
            z = z[:, self.permutations[layer_idx]]
            
            # Get scale and shift parameters from neural network
            params = self.conditioners[layer_idx](z)  # [batch, dim * 2]
            params = params.view(z.size(0), self.dim, 2)  # [batch, dim, 2]
            
            scales = params[:, :, 0]  # [batch, dim]
            shifts = params[:, :, 1]  # [batch, dim]
            
            # Stabilize scale parameters using tanh
            scales = torch.tanh(scales) * 2.0  # Scale between -2 and 2
            
            # Apply autoregressive transformation
            z_new = z.clone()
            for d in range(1, self.dim):  # Skip first dimension (autoregressive property)
                # Neural autoregressive transformation with improved stability
                scale = scales[:, d]
                shift = shifts[:, d]
                
                # Apply transformation: z_d = z_d * exp(scale) + shift
                z_new[:, d] = z[:, d] * torch.exp(scale) + shift
                log_det_total = log_det_total + scale  # Add log determinant contribution
            
            z = z_new
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        x = z.clone()
        
        for layer_idx in reversed(range(self.n_layers)):
            # Get parameters using current x
            params = self.conditioners[layer_idx](x)
            params = params.view(x.size(0), self.dim, 2)
            
            scales = torch.tanh(params[:, :, 0]) * 2.0
            shifts = params[:, :, 1]
            
            # Apply inverse transformation
            x_new = x.clone()
            for d in reversed(range(1, self.dim)):
                scale = scales[:, d]
                shift = shifts[:, d]
                # Inverse: x_d = (z_d - shift) * exp(-scale)
                x_new[:, d] = (x[:, d] - shift) * torch.exp(-scale)
            
            x = x_new
            
            # Apply inverse permutation
            inv_perm = torch.argsort(self.permutations[layer_idx])
            x = x[:, inv_perm]
        
        return x