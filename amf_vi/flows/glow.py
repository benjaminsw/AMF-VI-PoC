import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base_flow import BaseFlow

class ActNorm(nn.Module):
    """Activation Normalization layer from Glow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False
    
    def forward(self, x):
        """Forward pass with data-dependent initialization."""
        if not self.initialized and self.training:
            # Initialize scale and bias based on first batch
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0) + 1e-6
                self.bias.data = -mean
                self.scale.data = 1.0 / std
                self.initialized = True
        
        return self.scale * x + self.bias
    
    def inverse(self, y):
        """Inverse transformation."""
        return (y - self.bias) / self.scale
    
    def log_det(self, x):
        """Log determinant of Jacobian."""
        return torch.log(torch.abs(self.scale)).sum().expand(x.size(0))

class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 convolution from Glow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Initialize with random rotation matrix
        w_init = torch.randn(dim, dim)
        w_init = torch.linalg.qr(w_init)[0]  # QR decomposition for orthogonal matrix
        self.weight = nn.Parameter(w_init)
    
    def forward(self, x):
        """Forward transformation."""
        return F.linear(x, self.weight)
    
    def inverse(self, y):
        """Inverse transformation."""
        weight_inv = torch.inverse(self.weight)
        return F.linear(y, weight_inv)
    
    def log_det(self, x):
        """Log determinant of Jacobian."""
        log_det = torch.logdet(self.weight)
        return log_det.expand(x.size(0))

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer from Glow."""
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Create mask (alternating)
        mask = torch.zeros(dim)
        mask[::2] = 1  # Every other dimension
        self.register_buffer('mask', mask)
        
        # Network to predict scale and translation
        masked_dim = int(mask.sum().item())
        unmasked_dim = dim - masked_dim
        
        self.nn = nn.Sequential(
            nn.Linear(masked_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, unmasked_dim * 2)  # scale and translation
        )
        
        # Initialize last layer to zeros for stable training
        nn.init.zeros_(self.nn[-1].weight)
        nn.init.zeros_(self.nn[-1].bias)
    
    def forward(self, x):
        """Forward transformation."""
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        # Get coupling input
        coupling_input = x_masked[:, self.mask.bool()]
        
        if coupling_input.size(1) > 0:
            # Predict scale and translation
            st = self.nn(coupling_input)
            s, t = st.chunk(2, dim=1)  # Split into scale and translation
            s = torch.tanh(s)  # Stabilize scale
            
            # Apply affine transformation
            unmasked_indices = ~self.mask.bool()
            y = x.clone()
            y[:, unmasked_indices] = x_unmasked[:, unmasked_indices] * torch.exp(s) + t
        else:
            y = x
            s = torch.zeros(x.size(0), 0, device=x.device)
        
        return y, s
    
    def inverse(self, y):
        """Inverse transformation."""
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)
        
        # Get coupling input
        coupling_input = y_masked[:, self.mask.bool()]
        
        if coupling_input.size(1) > 0:
            # Predict scale and translation
            st = self.nn(coupling_input)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            
            # Apply inverse affine transformation
            unmasked_indices = ~self.mask.bool()
            x = y.clone()
            x[:, unmasked_indices] = (y_unmasked[:, unmasked_indices] - t) * torch.exp(-s)
        else:
            x = y
        
        return x

class GlowStep(nn.Module):
    """Single step of Glow flow."""
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actnorm = ActNorm(dim)
        self.invconv = InvertibleConv1x1(dim)
        self.coupling = AffineCouplingLayer(dim, hidden_dim)
    
    def forward(self, x):
        """Forward pass through Glow step."""
        # ActNorm
        x = self.actnorm.forward(x)
        log_det_actnorm = self.actnorm.log_det(x)
        
        # Invertible 1x1 conv
        x = self.invconv.forward(x)
        log_det_conv = self.invconv.log_det(x)
        
        # Affine coupling
        x, s = self.coupling.forward(x)
        log_det_coupling = s.sum(dim=1)
        
        total_log_det = log_det_actnorm + log_det_conv + log_det_coupling
        return x, total_log_det
    
    def inverse(self, z):
        """Inverse pass through Glow step."""
        # Reverse order
        z = self.coupling.inverse(z)
        z = self.invconv.inverse(z)
        z = self.actnorm.inverse(z)
        return z

class GlowFlow(BaseFlow):
    """Glow: Generative Flow with Invertible 1x1 Convolutions."""
    
    def __init__(self, dim: int, n_steps: int = 4, hidden_dim: int = 64):
        super().__init__(dim)
        self.n_steps = n_steps
        
        # Create flow steps with alternating masks
        self.steps = nn.ModuleList()
        for i in range(n_steps):
            step = GlowStep(dim, hidden_dim)
            # Alternate coupling masks
            if i % 2 == 1:  
                step.coupling.mask = 1 - step.coupling.mask
            self.steps.append(step)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Glow."""
        z = x
        total_log_det = torch.zeros(x.size(0), device=x.device)
        
        for step in self.steps:
            z, log_det = step.forward(z)
            total_log_det += log_det
        
        return z, total_log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x)
        return x