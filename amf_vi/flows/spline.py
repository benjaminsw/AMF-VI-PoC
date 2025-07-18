import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base_flow import BaseFlow

def searchsorted(bin_locations, inputs, eps=1e-6):
    """Find which bin each input belongs to."""
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

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
        """Apply rational quadratic spline transformation."""
        # Ensure inputs is 1D
        original_shape = inputs.shape
        inputs = inputs.view(-1)
        batch_size = inputs.size(0)
        
        # Parse parameters: widths, heights, derivatives
        num_bins = self.num_bins
        # params shape: [batch_size, 3*num_bins + 1]
        unnormalized_widths = params[..., :num_bins]
        unnormalized_heights = params[..., num_bins:2*num_bins]
        unnormalized_derivatives = params[..., 2*num_bins:]
        
        # For inputs outside tail bounds, use linear transformation
        inside_interval_mask = (inputs >= -self.tail_bound) & (inputs <= self.tail_bound)
        outside_interval_mask = ~inside_interval_mask
        
        outputs = torch.zeros_like(inputs)
        log_abs_det = torch.zeros_like(inputs)
        
        # Handle inputs outside the interval with linear transformation
        if outside_interval_mask.any():
            # Get first derivative for constant slope
            first_derivative = F.softplus(unnormalized_derivatives[..., 0]) + self.min_derivative
            outputs[outside_interval_mask] = inputs[outside_interval_mask]
            log_abs_det[outside_interval_mask] = torch.log(first_derivative[outside_interval_mask])
        
        if inside_interval_mask.any():
            inputs_inside = inputs[inside_interval_mask]
            params_inside = params[inside_interval_mask]
            
            # Process parameters for inside inputs only
            unnorm_w_inside = params_inside[..., :num_bins]
            unnorm_h_inside = params_inside[..., num_bins:2*num_bins]
            unnorm_d_inside = params_inside[..., 2*num_bins:]
            
            widths = F.softmax(unnorm_w_inside, dim=-1) * 2 * self.tail_bound
            heights = F.softmax(unnorm_h_inside, dim=-1) * 2 * self.tail_bound
            derivatives = F.softplus(unnorm_d_inside) + self.min_derivative
            
            # Ensure minimum width and height
            widths = widths * (2 * self.tail_bound - num_bins * self.min_bin_width) + self.min_bin_width
            heights = heights * (2 * self.tail_bound - num_bins * self.min_bin_height) + self.min_bin_height
            
            # Compute cumulative widths and heights (bin boundaries)
            cumwidths = torch.cumsum(widths, dim=-1)
            cumwidths = F.pad(cumwidths, (1, 0), mode='constant', value=-self.tail_bound)
            cumwidths[..., -1] = self.tail_bound
            
            cumheights = torch.cumsum(heights, dim=-1)
            cumheights = F.pad(cumheights, (1, 0), mode='constant', value=-self.tail_bound)
            cumheights[..., -1] = self.tail_bound
            
            # Find which bin each input belongs to
            bin_indices = searchsorted(cumwidths, inputs_inside)
            
            # Get bin boundaries for each input
            input_cumwidths = cumwidths.gather(-1, bin_indices[..., None])
            input_cumwidths_next = cumwidths.gather(-1, (bin_indices + 1)[..., None])
            input_cumheights = cumheights.gather(-1, bin_indices[..., None])
            input_cumheights_next = cumheights.gather(-1, (bin_indices + 1)[..., None])
            
            input_derivatives = derivatives.gather(-1, bin_indices[..., None])
            input_derivatives_next = derivatives.gather(-1, (bin_indices + 1)[..., None])
            
            # Compute bin widths and heights
            input_bin_widths = input_cumwidths_next - input_cumwidths
            input_bin_heights = input_cumheights_next - input_cumheights
            
            # Normalize input position within the bin
            theta = (inputs_inside[..., None] - input_cumwidths) / input_bin_widths
            theta = theta.squeeze(-1)
            
            if not inverse:
                # Forward transformation
                # Rational quadratic spline formula
                numerator = input_bin_heights * (input_derivatives * theta[..., None]**2 + 
                                               input_derivatives_next * theta[..., None] * (1 - theta[..., None]))
                denominator = input_derivatives * theta[..., None]**2 + 2 * input_derivatives_next * theta[..., None] * (1 - theta[..., None]) + (1 - theta[..., None])**2
                
                outputs_inside = input_cumheights.squeeze(-1) + (numerator / denominator).squeeze(-1)
                
                # Compute log absolute determinant of Jacobian
                derivative_numerator = (input_derivatives_next * input_derivatives * input_bin_heights * 
                                      denominator**2)
                
                log_abs_det_inside = torch.log(derivative_numerator.squeeze(-1) / 
                                             (input_bin_widths.squeeze(-1) * denominator.squeeze(-1)**2))
                
            else:
                # Inverse transformation - solve rational quadratic equation
                y_rel = (inputs_inside[..., None] - input_cumheights) / input_bin_heights
                y_rel = y_rel.squeeze(-1)
                
                # Quadratic formula coefficients
                a = input_bin_heights * (input_derivatives - input_derivatives_next) + y_rel[..., None] * (input_derivatives_next - input_derivatives)
                b = input_bin_heights * input_derivatives_next - y_rel[..., None] * (input_derivatives + input_derivatives_next)
                c = -input_derivatives * y_rel[..., None]
                
                # Solve quadratic equation
                discriminant = b**2 - 4 * a * c
                theta = 2 * c / (-b - torch.sqrt(discriminant))
                theta = theta.squeeze(-1)
                
                outputs_inside = input_cumwidths.squeeze(-1) + theta * input_bin_widths.squeeze(-1)
                
                # Compute log absolute determinant (inverse)
                denominator = input_derivatives * theta[..., None]**2 + 2 * input_derivatives_next * theta[..., None] * (1 - theta[..., None]) + (1 - theta[..., None])**2
                derivative_numerator = input_derivatives_next * input_derivatives * input_bin_heights * denominator**2
                
                log_abs_det_inside = -torch.log(derivative_numerator.squeeze(-1) / 
                                              (input_bin_widths.squeeze(-1) * denominator.squeeze(-1)**2))
            
            outputs[inside_interval_mask] = outputs_inside
            log_abs_det[inside_interval_mask] = log_abs_det_inside
        
        # Restore original shape
        outputs = outputs.view(original_shape)
        log_abs_det = log_abs_det.view(original_shape)
        
        return outputs, log_abs_det.sum() if log_abs_det.numel() > 1 else log_abs_det

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
                nn.Linear(hidden_dim, output_dim)
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
                log_det_layer = torch.zeros(x.size(0), device=x.device)
                
                for j in range(n_target):
                    start_idx = j * params_per_dim
                    end_idx = (j + 1) * params_per_dim
                    dim_params = spline_params[:, start_idx:end_idx]
                    
                    # Apply spline transformation
                    transformed_dim, log_det_dim = self.splines[i](
                        target_input[:, j], dim_params, inverse=False
                    )
                    
                    transformed_target[:, j] = transformed_dim
                    log_det_layer = log_det_layer + log_det_dim
                
                # Reconstruct z
                z_new = z.clone()
                z_new[:, target_indices] = transformed_target
                z = z_new
                
                log_det_total = log_det_total + log_det_layer
        
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
                        target_input[:, j], dim_params, inverse=True
                    )
                    
                    transformed_target[:, j] = transformed_dim
                
                # Reconstruct x
                x_new = x.clone()
                x_new[:, target_indices] = transformed_target
                x = x_new
        
        return x