import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base_flow import BaseFlow

class LinearTransformation(nn.Module):
    """Linear transformation layer for TAN."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Initialize as identity matrix with small perturbation
        weight = torch.eye(dim) + 0.01 * torch.randn(dim, dim)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        """Forward transformation: y = Wx + b"""
        return F.linear(x, self.weight, self.bias)
    
    def inverse(self, y):
        """Inverse transformation: x = W^{-1}(y - b)"""
        weight_inv = torch.inverse(self.weight)
        return F.linear(y - self.bias, weight_inv)
    
    def log_det(self, x):
        """Log determinant of Jacobian."""
        log_det = torch.logdet(self.weight)
        return log_det.expand(x.size(0))

class RecurrentTransformation(nn.Module):
    """RNN-based transformation from TAN."""
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Parameters for leaky ReLU transformation
        self.y = nn.Parameter(torch.ones(1))  # Scale parameter
        self.w = nn.Parameter(torch.randn(hidden_dim))
        self.b = nn.Parameter(torch.zeros(1))
        
        # RNN parameters for hidden state
        self.u = nn.Parameter(torch.ones(1))
        self.v = nn.Parameter(torch.randn(hidden_dim))
        self.a = nn.Parameter(torch.zeros(1))
        
        self.alpha = 0.1  # Leaky ReLU parameter
    
    def leaky_relu_forward(self, x, s):
        """Forward leaky ReLU transformation."""
        linear_part = self.y * x + torch.dot(self.w, s) + self.b
        return torch.where(linear_part >= 0, linear_part, self.alpha * linear_part)
    
    def leaky_relu_inverse(self, z, s):
        """Inverse leaky ReLU transformation."""
        linear_part = z - torch.dot(self.w, s) - self.b
        return torch.where(z >= torch.dot(self.w, s) + self.b, 
                          linear_part / self.y,
                          linear_part / (self.alpha * self.y))
    
    def leaky_relu_log_det(self, x, s):
        """Log determinant for leaky ReLU."""
        linear_part = self.y * x + torch.dot(self.w, s) + self.b
        mask = (linear_part >= 0).float()
        return torch.log(torch.abs(self.y)) + torch.log(mask + self.alpha * (1 - mask))
    
    def update_hidden_state(self, x, s_prev):
        """Update hidden state: s = ReLU(ux + v^T s_{prev} + a)"""
        return F.relu(self.u * x + torch.dot(self.v, s_prev) + self.a)
    
    def forward(self, x):
        """Forward transformation through RNN."""
        batch_size = x.size(0)
        z = torch.zeros_like(x)
        log_det_total = torch.zeros(batch_size, device=x.device)
        
        # Initialize hidden state
        s = torch.zeros(self.hidden_dim, device=x.device)
        
        for i in range(self.dim):
            # Transform current dimension
            z[:, i] = self.leaky_relu_forward(x[:, i], s)
            
            # Accumulate log determinant
            log_det_i = self.leaky_relu_log_det(x[:, i], s)
            log_det_total += log_det_i
            
            # Update hidden state for next dimension
            s = self.update_hidden_state(x[:, i], s)
        
        return z, log_det_total
    
    def inverse(self, z):
        """Inverse transformation through RNN."""
        batch_size = z.size(0)
        x = torch.zeros_like(z)
        
        # Initialize hidden state
        s = torch.zeros(self.hidden_dim, device=z.device)
        
        for i in range(self.dim):
            # Inverse transform current dimension
            x[:, i] = self.leaky_relu_inverse(z[:, i], s)
            
            # Update hidden state using recovered x
            s = self.update_hidden_state(x[:, i], s)
        
        return x

class TANFlow(BaseFlow):
    """Transformation Autoregressive Networks (TAN)."""
    
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 64, use_linear: bool = True):
        super().__init__(dim)
        self.n_layers = n_layers
        self.use_linear = use_linear
        
        # Create transformation layers
        self.transformations = nn.ModuleList()
        
        for i in range(n_layers):
            if use_linear and i % 2 == 0:
                # Alternate between linear and RNN transformations
                self.transformations.append(LinearTransformation(dim))
            else:
                self.transformations.append(RecurrentTransformation(dim, hidden_dim))
        
        # Permutation layers to increase expressiveness
        self.permutations = []
        for i in range(n_layers):
            if i == 0:
                perm = torch.arange(dim)
            else:
                perm = torch.randperm(dim)
            self.register_buffer(f'perm_{i}', perm)
            self.permutations.append(perm)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through TAN layers."""
        z = x
        total_log_det = torch.zeros(x.size(0), device=x.device)
        
        for i, transform in enumerate(self.transformations):
            # Apply permutation
            z = z[:, self.permutations[i]]
            
            if isinstance(transform, LinearTransformation):
                # Linear transformation
                z = transform.forward(z)
                log_det = transform.log_det(z)
            else:
                # RNN transformation
                z, log_det = transform.forward(z)
            
            total_log_det += log_det
        
        return z, total_log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        x = z
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(len(self.transformations))):
            transform = self.transformations[i]
            
            if isinstance(transform, LinearTransformation):
                x = transform.inverse(x)
            else:
                x = transform.inverse(x)
            
            # Apply inverse permutation
            inv_perm = torch.argsort(self.permutations[i])
            x = x[:, inv_perm]
        
        return x