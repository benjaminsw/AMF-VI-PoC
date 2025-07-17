import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow

class AutoregressiveNetwork(nn.Module):
    """Autoregressive network for IAF."""
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Create autoregressive networks for each dimension
        # Each output i depends only on inputs 0, 1, ..., i-1
        self.networks = nn.ModuleList()
        
        for i in range(dim):
            if i == 0:
                # First dimension has no dependencies, use constant
                net = nn.Sequential(
                    nn.Linear(1, hidden_dim),  # Dummy input
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2)  # scale and translation
                )
            else:
                # Dimension i depends on dimensions 0, ..., i-1
                net = nn.Sequential(
                    nn.Linear(i, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2)  # scale and translation
                )
            self.networks.append(net)
    
    def forward(self, x):
        """Compute scale and translation for each dimension."""
        batch_size = x.size(0)
        scales = torch.zeros(batch_size, self.dim, device=x.device)
        translations = torch.zeros(batch_size, self.dim, device=x.device)
        
        for i in range(self.dim):
            if i == 0:
                # First dimension uses dummy input (constant parameters)
                dummy_input = torch.ones(batch_size, 1, device=x.device)
                output = self.networks[i](dummy_input)
            else:
                # Use previous dimensions as input
                prev_dims = x[:, :i]
                output = self.networks[i](prev_dims)
            
            # Split output into scale and translation
            scales[:, i] = output[:, 0]
            translations[:, i] = output[:, 1]
        
        # Clamp scales to prevent numerical issues
        scales = torch.clamp(scales, min=-5, max=5)
        
        return scales, translations

class IAFFlow(BaseFlow):
    """Inverse Autoregressive Flow."""
    
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Create autoregressive networks for each layer
        self.autoregressive_nets = nn.ModuleList([
            AutoregressiveNetwork(dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Create permutations to increase expressiveness
        self.permutations = []
        for i in range(n_layers):
            if i == 0:
                perm = torch.arange(dim)  # Identity for first layer
            else:
                perm = torch.randperm(dim)  # Random permutation for others
            self.register_buffer(f'perm_{i}', perm)
            self.permutations.append(perm)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x -> z
        In IAF, this is the expensive direction (sequential computation)
        """
        z = x.clone()
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            # Apply permutation
            z = z[:, self.permutations[i]]
            
            # Compute scale and translation autoregressively
            batch_size = z.size(0)
            z_new = torch.zeros_like(z)
            
            for j in range(self.dim):
                if j == 0:
                    # First dimension uses dummy input
                    dummy_input = torch.ones(batch_size, 1, device=x.device)
                    net_output = self.autoregressive_nets[i].networks[j](dummy_input)
                    scale = net_output[:, 0]
                    translation = net_output[:, 1]
                else:
                    # Use current z values up to dimension j-1
                    prev_z = z_new[:, :j]
                    net_output = self.autoregressive_nets[i].networks[j](prev_z)
                    scale = net_output[:, 0]
                    translation = net_output[:, 1]
                
                # Clamp scale
                scale = torch.clamp(scale, min=-3, max=3)
                
                # Apply transformation: z_j = x_j * exp(s_j) + t_j
                z_new[:, j] = z[:, j] * torch.exp(scale) + translation
                log_det_total += scale
            
            z = z_new
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z -> x  
        In IAF, this is the efficient direction (parallel computation)
        """
        x = z.clone()
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(self.n_layers)):
            # Get scale and translation using autoregressive network
            scales, translations = self.autoregressive_nets[i](x)
            
            # Apply inverse transformation: x_j = (z_j - t_j) / exp(s_j)
            x = (x - translations) * torch.exp(-scales)
            
            # Apply inverse permutation
            inv_perm = torch.argsort(self.permutations[i])
            x = x[:, inv_perm]
        
        return x
    
    def _efficient_inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        More efficient inverse implementation that leverages IAF's strength.
        This version can compute all dimensions in parallel.
        """
        x = z.clone()
        
        for i in reversed(range(self.n_layers)):
            # Compute all scales and translations at once
            scales, translations = self.autoregressive_nets[i](x)
            
            # Apply inverse transformation in parallel
            x = (x - translations) * torch.exp(-scales)
            
            # Apply inverse permutation
            inv_perm = torch.argsort(self.permutations[i])
            x = x[:, inv_perm]
        
        return x