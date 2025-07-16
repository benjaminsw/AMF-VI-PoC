import torch
import torch.nn as nn
from typing import List, Dict, Any
from .flows.realnvp import RealNVPFlow
from .flows.planar import PlanarFlow
from .flows.radial import RadialFlow

class SimpleAMFVI(nn.Module):
    """Simplified AMF-VI model for PoC."""
    
    def __init__(self, dim: int = 2, flow_types: List[str] = None, n_components: int = 3):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        
        if flow_types is None:
            flow_types = ['realnvp', 'planar', 'radial']
        
        # Create different flow types
        self.flows = nn.ModuleList()
        for i, flow_type in enumerate(flow_types[:n_components]):
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=4))
            elif flow_type == 'planar':
                self.flows.append(PlanarFlow(dim, n_layers=8))
            elif flow_type == 'radial':
                self.flows.append(RadialFlow(dim, n_layers=8))
            else:
                raise ValueError(f"Unknown flow type: {flow_type}")
        
        # Simple gating network (mixture weights)
        self.gating_net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.flows)),
            nn.Softmax(dim=1)
        )
        
        # Mode separation regularization weight
        self.reg_weight = 0.01
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all flows."""
        batch_size = x.size(0)
        
        # Get mixture weights
        weights = self.gating_net(x)
        
        # Apply each flow
        z_list = []
        log_det_list = []
        log_prob_list = []
        
        for flow in self.flows:
            z, log_det = flow.forward_and_log_det(x)
            log_prob = flow.log_prob(x)
            
            z_list.append(z)
            log_det_list.append(log_det)
            log_prob_list.append(log_prob)
        
        return {
            'z': torch.stack(z_list, dim=1),  # [batch, n_flows, dim]
            'log_det': torch.stack(log_det_list, dim=1),  # [batch, n_flows]
            'log_prob': torch.stack(log_prob_list, dim=1),  # [batch, n_flows]
            'weights': weights,  # [batch, n_flows]
        }
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the mixture."""
        # Sample component assignments
        component_probs = torch.ones(len(self.flows)) / len(self.flows)
        components = torch.multinomial(component_probs, n_samples, replacement=True)
        
        samples = []
        for i, comp_idx in enumerate(components):
            sample = self.flows[comp_idx].sample(1)
            samples.append(sample)
        
        return torch.cat(samples, dim=0)
    
    def log_prob_mixture(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mixture log probability."""
        output = self.forward(x)
        
        # Weighted log probabilities
        weighted_log_probs = output['log_prob'] + torch.log(output['weights'] + 1e-8)
        
        # Log-sum-exp for mixture
        return torch.logsumexp(weighted_log_probs, dim=1)
    
    def regularization_loss(self) -> torch.Tensor:
        """Simple mode separation regularization."""
        if len(self.flows) < 2:
            return torch.tensor(0.0)
        
        # Sample from each flow and encourage diversity
        n_samples = 50
        samples = []
        for flow in self.flows:
            flow_samples = flow.sample(n_samples)
            samples.append(flow_samples.mean(dim=0))  # Mean of each flow
        
        # Encourage different means
        total_loss = torch.tensor(0.0)
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = torch.norm(samples[i] - samples[j])
                total_loss += torch.exp(-dist)  # Penalize close means
        
        return self.reg_weight * total_loss