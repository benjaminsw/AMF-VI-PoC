import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow
from scipy.stats import norm

class TrainableKernelLayer(nn.Module):
    """Trainable marginal Gaussianization using mixture of logistics."""
    
    def __init__(self, dim: int, n_anchors: int = 20):
        super().__init__()
        self.dim = dim
        self.n_anchors = n_anchors
        
        # Learnable anchor points and bandwidths for each dimension
        self.anchors = nn.Parameter(torch.randn(dim, n_anchors))  # [dim, n_anchors]
        self.bandwidths = nn.Parameter(torch.ones(dim, n_anchors) * 0.1)  # [dim, n_anchors]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply marginal Gaussianization to each dimension."""
        batch_size = x.size(0)
        z_list = []
        log_det = torch.zeros(batch_size, device=x.device)
        
        for d in range(self.dim):
            # Compute CDF using mixture of logistics: F(x) = (1/K) * sum(sigmoid((x - mu_k) / h_k))
            x_d = x[:, d].unsqueeze(1)  # [batch, 1]
            anchors_d = self.anchors[d].unsqueeze(0)  # [1, n_anchors]
            bandwidths_d = torch.clamp(self.bandwidths[d].unsqueeze(0), min=1e-6)  # [1, n_anchors]
            
            # Compute sigmoid terms
            sigmoid_terms = torch.sigmoid((x_d - anchors_d) / bandwidths_d)  # [batch, n_anchors]
            cdf_values = sigmoid_terms.mean(dim=1)  # [batch]
            
            # Clamp CDF values to avoid extreme inverse Gaussian CDF values
            cdf_values = torch.clamp(cdf_values, min=1e-6, max=1-1e-6)
            
            # Apply inverse Gaussian CDF (approximation using torch.erfinv)
            # Φ^(-1)(u) ≈ √2 * erfinv(2u - 1)
            z_d = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.erfinv(2 * cdf_values - 1)
            z_list.append(z_d)
            
            # Compute log determinant: log|dΨ/dx| = log(φ(Ψ(x))) - log(f(x))
            # where φ is standard Gaussian PDF and f is the derivative of our CDF
            
            # Gaussian PDF at transformed point
            gaussian_pdf = torch.exp(-0.5 * z_d**2) / torch.sqrt(torch.tensor(2 * torch.pi, device=x.device))
            
            # Derivative of our CDF
            sigmoid_derivs = torch.sigmoid((x_d - anchors_d) / bandwidths_d) * (1 - torch.sigmoid((x_d - anchors_d) / bandwidths_d))
            sigmoid_derivs = sigmoid_derivs / bandwidths_d  # Chain rule
            cdf_derivative = sigmoid_derivs.mean(dim=1)  # [batch]
            cdf_derivative = torch.clamp(cdf_derivative, min=1e-8)
            
            # Log determinant for this dimension
            log_det += torch.log(gaussian_pdf) - torch.log(cdf_derivative)
        
        z = torch.stack(z_list, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Approximate inverse using numerical methods."""
        # For simplicity, use iterative method (not efficient but works for testing)
        x = z.clone()
        
        for _ in range(5):  # Limited iterations
            try:
                z_pred, _ = self.forward(x)
                x = x - 0.1 * (z_pred - z)  # Simple gradient step
            except:
                break
        
        return x

class HouseholderReflection(nn.Module):
    """Trainable orthogonal transformation using Householder reflections."""
    
    def __init__(self, dim: int, n_reflections: int = None):
        super().__init__()
        self.dim = dim
        self.n_reflections = n_reflections or dim
        
        # Learnable reflection vectors
        self.reflection_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(dim)) for _ in range(self.n_reflections)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sequence of Householder reflections."""
        z = x.clone()
        
        for v_param in self.reflection_vectors:
            # Normalize reflection vector
            v = v_param / (torch.norm(v_param) + 1e-8)
            
            # Apply Householder reflection: z = z - 2 * (v^T z) v
            vt_z = torch.sum(z * v.unsqueeze(0), dim=1, keepdim=True)  # [batch, 1]
            z = z - 2 * vt_z * v.unsqueeze(0)  # [batch, dim]
        
        # Log determinant is 0 for orthogonal transformations
        log_det = torch.zeros(x.size(0), device=x.device)
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse is the transpose (applying reflections in reverse order)."""
        x = z.clone()
        
        # Apply reflections in reverse order
        for v_param in reversed(self.reflection_vectors):
            v = v_param / (torch.norm(v_param) + 1e-8)
            vt_x = torch.sum(x * v.unsqueeze(0), dim=1, keepdim=True)
            x = x - 2 * vt_x * v.unsqueeze(0)
        
        return x

class GaussianizationFlow(BaseFlow):
    """Gaussianization Flow: alternating marginal Gaussianization and rotation."""
    
    def __init__(self, dim: int, n_layers: int = 4, n_anchors: int = 20, n_reflections: int = None):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Create alternating layers
        self.kernel_layers = nn.ModuleList([
            TrainableKernelLayer(dim, n_anchors) for _ in range(n_layers)
        ])
        
        self.rotation_layers = nn.ModuleList([
            HouseholderReflection(dim, n_reflections) for _ in range(n_layers)
        ])
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x -> z with log determinant."""
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            # Apply marginal Gaussianization
            z, log_det_kernel = self.kernel_layers[i](z)
            log_det_total += log_det_kernel
            
            # Apply rotation
            z, log_det_rotation = self.rotation_layers[i](z)
            log_det_total += log_det_rotation  # Should be 0 for orthogonal transforms
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z -> x."""
        x = z
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(self.n_layers)):
            # Inverse rotation
            x = self.rotation_layers[i].inverse(x)
            
            # Inverse marginal Gaussianization
            x = self.kernel_layers[i].inverse(x)
        
        return x