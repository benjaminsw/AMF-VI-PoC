"""
KDE-based KL Divergence Computation using Gaussian Kernel Density Estimation.

This module provides functions to compute KL divergence between two sets of samples
using Kernel Density Estimation instead of histogram-based approximation.
"""

import torch
import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, Union

def compute_kde_kl_divergence(target_samples: torch.Tensor, 
                            generated_samples: torch.Tensor,
                            grid_resolution: int = 100,
                            bandwidth_method: str = 'scott',
                            epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between target and generated samples using KDE.
    
    Args:
        target_samples: torch.Tensor [n_samples, 2] - target distribution samples
        generated_samples: torch.Tensor [n_samples, 2] - generated distribution samples  
        grid_resolution: int - number of grid points per dimension (100x100 or 200x200)
        bandwidth_method: str - bandwidth selection method ('scott', 'silverman', or float)
        epsilon: float - small value to avoid log(0) issues
        
    Returns:
        float - KL divergence D_KL(P||Q) where P=target, Q=generated
    """
    # Step 1: Estimate Densities using scipy.stats.gaussian_kde
    target_np = target_samples.detach().cpu().numpy().T  # Shape: [2, n_samples]
    generated_np = generated_samples.detach().cpu().numpy().T  # Shape: [2, n_samples]
    
    # Fit KDE to both distributions
    # Bandwidth Selection Strategy - Option A: Use same bandwidth for fair comparison
    if isinstance(bandwidth_method, str):
        kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
        kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
        
        # Use same bandwidth for fair comparison
        target_bandwidth = kde_target.factor
        kde_generated.set_bandwidth(target_bandwidth)
    else:
        # Custom bandwidth
        kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
        kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
    
    # Step 2: Create Evaluation Grid
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    # Add small margin to ensure all data is covered
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])  # Shape: [2, grid_resolution^2]
    
    # Step 3: Evaluate KDE Densities on Grid
    p_densities = kde_target(grid_points)  # Target distribution P
    q_densities = kde_generated(grid_points)  # Generated distribution Q
    
    # Step 4: Numerical Integration for KL Divergence
    # Add epsilon to avoid log(0) issues
    p_densities = np.maximum(p_densities, epsilon)
    q_densities = np.maximum(q_densities, epsilon)
    
    # Compute KL divergence: D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
    log_ratio = np.log(p_densities / q_densities)
    kl_integrand = p_densities * log_ratio
    
    # Grid cell area for numerical integration
    dx = (x_max - x_min) / grid_resolution
    dy = (y_max - y_min) / grid_resolution
    cell_area = dx * dy
    
    # Numerical integration using Riemann sum
    kl_divergence = np.sum(kl_integrand) * cell_area
    
    return float(kl_divergence)

def compute_kde_kl_divergence_with_bandwidth_options(target_samples: torch.Tensor,
                                                   generated_samples: torch.Tensor,
                                                   bandwidth_strategy: str = 'same',
                                                   grid_resolution: int = 100) -> float:
    """
    Compute KL divergence with different bandwidth selection strategies.
    
    Args:
        target_samples: torch.Tensor [n_samples, 2] - target distribution samples
        generated_samples: torch.Tensor [n_samples, 2] - generated distribution samples
        bandwidth_strategy: str - bandwidth selection strategy
            - 'same': Use same bandwidth for both KDEs (fair comparison) 
            - 'separate': Optimize bandwidth separately (may be more accurate)
            - 'cross_val': Use cross-validation to select optimal bandwidth (not implemented)
        grid_resolution: int - grid resolution per dimension
        
    Returns:
        float - KL divergence
    """
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    if bandwidth_strategy == 'same':
        # Option A: Use same bandwidth for both KDEs (fair comparison)
        kde_target = gaussian_kde(target_np, bw_method='scott')
        kde_generated = gaussian_kde(generated_np, bw_method='scott')
        
        # Use target's bandwidth for both
        target_bandwidth = kde_target.factor
        kde_generated.set_bandwidth(target_bandwidth)
        
    elif bandwidth_strategy == 'separate':
        # Option B: Optimize bandwidth separately (may be more accurate)
        kde_target = gaussian_kde(target_np, bw_method='scott')
        kde_generated = gaussian_kde(generated_np, bw_method='scott')
        
    elif bandwidth_strategy == 'cross_val':
        # Option C: Use cross-validation to select optimal bandwidth
        # TODO: Implement cross-validation bandwidth selection
        raise NotImplementedError("Cross-validation bandwidth selection not yet implemented")
        
    else:
        raise ValueError(f"Unknown bandwidth strategy: {bandwidth_strategy}")
    
    # Create evaluation grid and compute KL divergence
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    p_densities = kde_target(grid_points)
    q_densities = kde_generated(grid_points)
    
    epsilon = 1e-10
    p_densities = np.maximum(p_densities, epsilon)
    q_densities = np.maximum(q_densities, epsilon)
    
    log_ratio = np.log(p_densities / q_densities)
    kl_integrand = p_densities * log_ratio
    
    dx = (x_max - x_min) / grid_resolution
    dy = (y_max - y_min) / grid_resolution
    cell_area = dx * dy
    
    kl_divergence = np.sum(kl_integrand) * cell_area
    return float(kl_divergence)

def adaptive_grid_resolution(target_samples: torch.Tensor,
                           generated_samples: torch.Tensor,
                           max_resolution: int = 200,
                           min_resolution: int = 50) -> int:
    """
    Adaptively choose grid resolution based on data characteristics.
    
    Args:
        target_samples: torch.Tensor - target samples
        generated_samples: torch.Tensor - generated samples  
        max_resolution: int - maximum grid resolution
        min_resolution: int - minimum grid resolution
        
    Returns:
        int - recommended grid resolution
    """
    n_target = target_samples.size(0)
    n_generated = generated_samples.size(0)
    
    # Base resolution on sample size
    total_samples = n_target + n_generated
    
    if total_samples < 500:
        return min_resolution
    elif total_samples < 2000:
        return 100
    else:
        return max_resolution

def get_kde_info(target_samples: torch.Tensor, 
                generated_samples: torch.Tensor) -> dict:
    """
    Get information about KDE fitting for debugging purposes.
    
    Args:
        target_samples: torch.Tensor - target samples
        generated_samples: torch.Tensor - generated samples
        
    Returns:
        dict - KDE information including bandwidths and sample statistics
    """
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    kde_target = gaussian_kde(target_np, bw_method='scott')
    kde_generated = gaussian_kde(generated_np, bw_method='scott')
    
    return {
        'target_bandwidth': kde_target.factor,
        'generated_bandwidth': kde_generated.factor,
        'target_n_samples': target_samples.size(0),
        'generated_n_samples': generated_samples.size(0),
        'target_mean': target_samples.mean(dim=0).tolist(),
        'generated_mean': generated_samples.mean(dim=0).tolist(),
        'target_std': target_samples.std(dim=0).tolist(),
        'generated_std': generated_samples.std(dim=0).tolist(),
    }