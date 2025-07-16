import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow

class RadialFlow(BaseFlow):
    """Radial normalizing flow: f(z) = z + β * h(α, r) * (z - z0)."""
    
    def __init__(self, dim: int, n_layers: int = 8):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Parameters for each layer
        self.z0 = nn.ParameterList([nn.Parameter(torch.randn(1, dim)) for _ in range(n_layers)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_layers)])
        self.beta = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_layers)])
    
    def h(self, alpha, r):
        """Activation function: h(α, r) = 1 / (α + r)."""
        return 1 / (alpha + r)
    
    def h_derivative(self, alpha, r):
        """Derivative of h with respect to r."""
        return -1 / (alpha + r)**2
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            # Ensure alpha > 0 for stability
            alpha = torch.exp(self.alpha[i])
            beta = self.beta[i]
            z0 = self.z0[i]
            
            # Compute distance from center
            diff = z - z0
            r = torch.norm(diff, dim=1, keepdim=True)
            
            # Compute h and its derivative
            h_val = self.h(alpha, r)
            h_prime = self.h_derivative(alpha, r)
            
            # Forward transformation
            z = z + beta * h_val * diff
            
            # Compute log determinant
            # |det J| = |1 + β * h'(α, r) + β * h(α, r) * (d-1)/r|
            det_term = 1 + beta * h_prime + beta * h_val * (self.dim - 1) / (r + 1e-8)
            log_det_total += torch.log(torch.abs(det_term.squeeze()) + 1e-8)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Numerical inverse for radial flow."""
        x = z.clone()
        
        # Fixed-point iteration
        for _ in range(10):
            z_pred, _ = self.forward_and_log_det(x)
            x = x - 0.1 * (z_pred - z)
        
        return x