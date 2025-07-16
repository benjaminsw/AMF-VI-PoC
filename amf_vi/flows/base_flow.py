import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

class BaseFlow(nn.Module, ABC):
    """Simplified base flow class for PoC."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    @abstractmethod
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x to z and compute log determinant."""
        pass
    
    @abstractmethod
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transform z back to x."""
        pass
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the flow."""
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.dim, device=device)
        return self.inverse(z)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        z, log_det = self.forward_and_log_det(x)
        log_prob_base = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * torch.log(2 * torch.pi)
        return log_prob_base + log_det