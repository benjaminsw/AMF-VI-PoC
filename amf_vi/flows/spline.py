import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base_flow import BaseFlow

class RationalQuadraticSpline(nn.Module):
    """Rational quadratic spline transformation."""
    
    def __init__(self, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3
    
    def forward(self, inputs, params, inverse=False):
        """Apply spline transformation."""
        batch_size = inputs.size(0)
        
        # For inputs outside tail bounds, use linear transformation
        inside_interval_mask = (inputs >= -self.tail_bound) & (inputs <= self.tail_bound)
        outside_interval_mask = ~inside_interval_mask
        
        outputs = torch.zeros_like(inputs)
        log_abs_det = torch.zeros(batch_size, device=inputs.device)
        
        # Handle inputs outside the interval with identity + small slope
        if outside_interval_mask.any():
            outputs[outside_interval_mask] = inputs[outside_interval_mask]
            # No change to log_abs_det for identity transform (log(1) = 0)
        
        if inside_interval_mask.any():
            # Extract inputs inside the interval
            inputs_inside = inputs[inside_interval_mask]
            
            if inputs_inside.numel() > 0:
                # For this PoC, use simple linear transformation
                # In practice, this would implement full rational quadratic splines
                scale = 1.0
                outputs[inside_interval_mask] = inputs_inside * scale
                
                # Find which batch elements have any inside interval inputs
                inside_batch_mask = inside_interval_mask.any(dim=1) if inside_interval_mask.dim() > 1 else inside_interval_mask
                
                # **FIX: Only update log_abs_det for elements that actually have inside interval inputs**
                if inside_batch_mask.any():
                    # Create log_det contribution for inside elements only
                    inside_log_det = torch.zeros(inside_batch_mask.sum().item(), device=inputs.device)
                    log_abs_det[inside_batch_mask] += inside_log_det
        
        return outputs, log_abs_det
    
    def _forward_spline(self, inputs, cumwidths, cumheights, derivatives):
        """Forward rational quadratic spline transformation."""
        # Simplified implementation using linear transformation
        batch_size = inputs.size(0)
        
        # Just apply a simple linear transformation for this PoC
        scale = 1.0
        outputs = inputs * scale
        log_abs_det = torch.zeros(batch_size, device=inputs.device)
        
        return outputs, log_abs_det
    
    def _inverse_spline(self, inputs, cumwidths, cumheights, derivatives):
        """Inverse rational quadratic spline transformation."""
        # Simplified implementation - just return the inputs for identity transform
        batch_size = inputs.size(0)
        log_abs_det = torch.zeros(batch_size, device=inputs.device)
        return inputs, log_abs_det

class SplineFlow(BaseFlow):
    """Neural Spline Flow using rational quadratic splines."""
    
    def __init__(self, dim: int, n_layers: int = 4, num_bins: int = 8, hidden_dim: int = 64):
        super().__init__(dim)
        self.n_layers = n_layers
        self.num_bins = num_bins
        
        # Create coupling layers
        self.transforms = nn.ModuleList()
        self.splines = nn.ModuleList()
        
        for i in range(n_layers):
            # Determine split
            if dim % 2 == 0:
                split_dim = dim // 2
            else:
                split_dim = dim // 2 + (i % 2)  # Alternate for odd dimensions
            
            input_dim = split_dim
            output_dim = (dim - split_dim) * (3 * num_bins + 1)
            
            # Coupling network to generate spline parameters
            transform_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()  # Bounded output for stability
            )
            
            self.transforms.append(transform_net)
            self.splines.append(RationalQuadraticSpline(num_bins))
        
        # Create alternating masks
        self.masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim//2] = 1
            else:
                mask[dim//2:] = 1
            self.register_buffer(f'mask_{i}', mask)
            self.masks.append(mask)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x.clone()
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            mask = self.masks[i].to(x.device)
            
            # Split into conditioner and target using proper indexing
            conditioner_indices = mask.bool()
            target_indices = ~mask.bool()
            
            # Get conditioner input by selecting columns
            conditioner_input = z[:, conditioner_indices]
            target_input = z[:, target_indices]
            
            if conditioner_input.numel() > 0 and target_input.numel() > 0:
                # Generate spline parameters from conditioner
                spline_params = self.transforms[i](conditioner_input)
                
                # Apply spline to target
                n_target = target_input.size(1)
                params_per_dim = spline_params.size(-1) // n_target
                
                transformed_target = torch.zeros_like(target_input)
                
                for j in range(n_target):
                    start_idx = j * params_per_dim
                    end_idx = (j + 1) * params_per_dim
                    dim_params = spline_params[:, start_idx:end_idx]
                    
                    # Apply spline transformation
                    transformed_dim, log_det_dim = self.splines[i](
                        target_input[:, j:j+1], dim_params, inverse=False
                    )
                    
                    transformed_target[:, j:j+1] = transformed_dim
                    log_det_total += log_det_dim
                
                # Reconstruct z
                z_new = z.clone()
                z_new[:, target_indices] = transformed_target
                z = z_new
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z.clone()
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(self.n_layers)):
            mask = self.masks[i].to(z.device)
            
            # Split using proper indexing
            conditioner_indices = mask.bool()
            target_indices = ~mask.bool()
            
            conditioner_input = x[:, conditioner_indices]
            target_input = x[:, target_indices]
            
            if conditioner_input.numel() > 0 and target_input.numel() > 0:
                # Generate spline parameters from conditioner
                spline_params = self.transforms[i](conditioner_input)
                
                # Apply inverse spline to target
                n_target = target_input.size(1)
                params_per_dim = spline_params.size(-1) // n_target
                
                transformed_target = torch.zeros_like(target_input)
                
                for j in range(n_target):
                    start_idx = j * params_per_dim
                    end_idx = (j + 1) * params_per_dim
                    dim_params = spline_params[:, start_idx:end_idx]
                    
                    transformed_dim, _ = self.splines[i](
                        target_input[:, j:j+1], dim_params, inverse=True
                    )
                    
                    transformed_target[:, j:j+1] = transformed_dim
                
                # Reconstruct x
                x_new = x.clone()
                x_new[:, target_indices] = transformed_target
                x = x_new
        
        return x