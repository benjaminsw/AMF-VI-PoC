"""
Simple data generator for AMF-VI synthetic datasets.
Usage: from data.data_generator import generate_data
"""

import torch
import numpy as np
from sklearn.datasets import make_moons


def create_banana_data(n_samples=1000, noise=0.1):
    """Banana: N(x2; x1^2/4, 1) * N(x1; 0, 2)"""
    x1 = np.random.normal(0, np.sqrt(2), n_samples)
    x2 = np.random.normal(x1**2 / 4, 1.0, n_samples)
    if noise > 0:
        x1 += np.random.normal(0, noise, n_samples)
        x2 += np.random.normal(0, noise, n_samples)
    return torch.tensor(np.column_stack([x1, x2]), dtype=torch.float32)


def create_x_shape_data(n_samples=1000, noise=0.1):
    """X-shape: Mixture of two diagonal Gaussians"""
    n_half = n_samples // 2
    cov1 = np.array([[2.0, 1.8], [1.8, 2.0]])
    cov2 = np.array([[2.0, -1.8], [-1.8, 2.0]])
    
    samples1 = np.random.multivariate_normal([0, 0], cov1, n_half)
    samples2 = np.random.multivariate_normal([0, 0], cov2, n_samples - n_half)
    
    data = np.vstack([samples1, samples2])
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_bimodal_shared_base(n_samples=1000, separation=3.0, noise=0.3):
    """Bimodal with shared covariance"""
    n_half = n_samples // 2
    cov = np.array([[0.5, 0.0], [0.0, 0.5]])
    
    samples1 = np.random.multivariate_normal([-separation/2, 0], cov, n_half)
    samples2 = np.random.multivariate_normal([separation/2, 0], cov, n_samples - n_half)
    
    data = np.vstack([samples1, samples2])
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_bimodal_different_base(n_samples=1000, separation=4.5, noise=0.2):
    """Bimodal with different covariances"""
    n_half = n_samples // 2
    cov1 = np.array([[0.8, 0.2], [0.2, 0.3]])
    cov2 = np.array([[0.3, -0.1], [-0.1, 0.6]])
    
    samples1 = np.random.multivariate_normal([-separation/2, -0.5], cov1, n_half)
    samples2 = np.random.multivariate_normal([separation/2, 0.5], cov2, n_samples - n_half)
    
    data = np.vstack([samples1, samples2])
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_multimodal_gaussian_mixture(n_samples=1000, n_modes=3, noise=0.1):
    """Multimodal Gaussian mixture"""
    modes = [(-2.0, -1.0), (2.0, 1.0), (0.0, 2.5)][:n_modes]
    samples_per_mode = n_samples // len(modes)
    
    all_samples = []
    for mode in modes:
        samples = np.random.multivariate_normal(mode, 0.3 * np.eye(2), samples_per_mode)
        all_samples.append(samples)
    
    data = np.vstack(all_samples)
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_two_moons_data(n_samples=1000, noise=0.1):
    """Two moons from sklearn"""
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(X, dtype=torch.float32)


def create_concentric_rings(n_samples=1000, noise=0.1):
    """Concentric rings"""
    n_half = n_samples // 2
    
    # Inner ring
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    r1 = np.random.normal(1.0, noise, n_half)
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    
    # Outer ring
    theta2 = np.random.uniform(0, 2*np.pi, n_samples - n_half)
    r2 = np.random.normal(2.5, noise, n_samples - n_half)
    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
    
    data = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)


# Registry for easy access
DATA_GENERATORS = {
    'banana': create_banana_data,
    'x_shape': create_x_shape_data,
    'bimodal_shared': create_bimodal_shared_base,
    'bimodal_different': create_bimodal_different_base,
    'multimodal': create_multimodal_gaussian_mixture,
    'two_moons': create_two_moons_data,
    'rings': create_concentric_rings
}


def generate_data(dataset_name, n_samples=1000, **kwargs):
    """
    Generate synthetic data by name.
    
    Args:
        dataset_name: Name from DATA_GENERATORS keys
        n_samples: Number of samples
        **kwargs: Additional parameters for specific generators
    
    Returns:
        torch.Tensor: Generated data [n_samples, 2]
    """
    if dataset_name not in DATA_GENERATORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATA_GENERATORS.keys())}")
    
    return DATA_GENERATORS[dataset_name](n_samples=n_samples, **kwargs)


def get_available_datasets():
    """Get list of available dataset names."""
    return list(DATA_GENERATORS.keys())