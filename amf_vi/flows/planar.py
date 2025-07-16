import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow

class PlanarFlow(BaseFlow):
    """Planar normalizing flow: f(z) = z + u * h(w^T z + b)."""
    
    def __init__(self, dim: int, n_layers: int = 8):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Parameters for each layer
        self.u = nn.ParameterList([nn.Parameter(torch.randn(1, dim)) for _ in range(n_layers)])
        self.w = nn.ParameterList([nn.Parameter(torch.randn(1, dim)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_layers)])
    
    def h(self, x):
        """Activation function: tanh."""
        return torch.tanh(x)
    
    def h_derivative(self, x):
        """Derivative of activation function."""
        return 1 - torch.tanh(x)**2
    
    def constrain_u(self, u, w):
        """Ensure invertibility by constraining u."""
        # u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
        # where m(x) = -1 + log(1 + exp(x))
        wtu = (w * u).sum(dim=1, keepdim=True)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        u_hat = u + (m_wtu - wtu) * w / (w**2).sum(dim=1, keepdim=True)
        return u_hat
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            u_hat = self.constrain_u(self.u[i], self.w[i])
            
            # Forward transformation
            linear = torch.mm(z, self.w[i].t()) + self.b[i]
            z = z + u_hat * self.h(linear)
            
            # Compute log determinant
            psi = self.h_derivative(linear) * self.w[i]
            det = 1 + torch.mm(psi, u_hat.t()).squeeze()
            log_det_total += torch.log(torch.abs(det) + 1e-8)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Note: Planar flows don't have analytical inverse. Use iterative method."""
        # For PoC, we'll use a simple approximation or numerical method
        # In practice, you might want to use more sophisticated inversion
        x = z.clone()
        
        # Simple fixed-point iteration (not guaranteed to converge)
        for _ in range(10):  # Limited iterations for speed
            z_pred, _ = self.forward_and_log_det(x)
            x = x - 0.1 * (z_pred - z)  # Simple gradient step
        
        return x