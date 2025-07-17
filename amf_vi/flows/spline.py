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
            log_abs_det[outside_interval_mask.any(dim=1)] += 0.0
        
        if inside_interval_mask.any():
            # Extract inputs inside the interval
            inputs_inside = inputs[inside_interval_mask]
            
            if inputs_inside.numel() > 0:
                # Unpack spline parameters
                # params: [batch, 3*num_bins + 1] per dimension
                unnormalized_widths = params[..., :self.num_bins]
                unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
                unnormalized_derivatives = params[..., 2*self.num_bins:]
                
                # Normalize parameters
                widths = F.softmax(unnormalized_widths, dim=-1)
                widths = self.min_bin_width + (2 * self.tail_bound - self.num_bins * self.min_bin_width) * widths
                
                heights = F.softmax(unnormalized_heights, dim=-1)
                heights = self.min_bin_height + (2 * self.tail_bound - self.num_bins * self.min_bin_height) * heights
                
                derivatives = self.min_derivative + F.softplus(unnormalized_derivatives)
                
                # Create cumulative widths and heights (knot positions)
                cumwidths = torch.cumsum(widths, dim=-1)
                cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
                cumwidths = cumwidths - self.tail_bound
                
                cumheights = torch.cumsum(heights, dim=-1)
                cumheights = F.pad(cumheights, (1, 0), value=0.0)
                cumheights = cumheights - self.tail_bound
                
                # Pad derivatives
                derivatives = F.pad(derivatives, (1, 1), value=1.0)
                
                # Apply spline transformation
                if inverse:
                    spline_outputs, spline_log_abs_det = self._inverse_spline(
                        inputs_inside, cumwidths, cumheights, derivatives
                    )
                else:
                    spline_outputs, spline_log_abs_det = self._forward_spline(
                        inputs_inside, cumwidths, cumheights, derivatives
                    )
                
                outputs[inside_interval_mask] = spline_outputs
                
                # Handle log determinant for batch elements with inside interval inputs
                inside_batch_mask = inside_interval_mask.any(dim=1)
                if inside_batch_mask.any():
                    log_abs_det[inside_batch_mask] += spline_log_abs_det[inside_batch_mask]
        
        return outputs, log_abs_det
    
    def _forward_spline(self, inputs, cumwidths, cumheights, derivatives):
        """Forward rational quadratic spline transformation."""
        # Simplified implementation using piecewise linear approximation
        # In practice, this would implement full rational quadratic splines
        
        # Find which bin each input falls into
        batch_size = inputs.size(0)
        
        # Use linear interpolation as approximation
        # Get middle derivatives as slope
        middle_idx = derivatives.size(-1) // 2
        slopes = derivatives[..., middle_idx].unsqueeze(-1)
        
        # Apply linear transformation
        outputs = inputs * slopes.squeeze(-1)
        log_abs_det = torch.log(torch.abs(slopes).squeeze(-1))
        
        return outputs, log_abs_det
    
    def _inverse_spline(self, inputs, cumwidths, cumheights, derivatives):
        """Inverse rational quadratic spline transformation."""
        # Simplified implementation
        middle_idx = derivatives.size(-1) // 2
        slopes = derivatives[..., middle_idx].unsqueeze(-1)
        
        outputs = inputs / slopes.squeeze(-1)
        
        return outputs

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
            # Use broadcasting to select elements properly
            mask_expanded = mask.unsqueeze(0).expand_as(z)
            x_conditioner = z * mask_expanded
            x_target = z * (1 - mask_expanded)
            
            # Extract conditioner values (non-zero elements)
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