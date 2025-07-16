import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from typing import Tuple, Callable

def create_multimodal_data(n_samples: int = 1000) -> torch.Tensor:
    """Create 2D multimodal Gaussian mixture data."""
    # Define 3 modes
    modes = [
        (-2.0, -1.0),
        (2.0, 1.0),
        (0.0, 2.5)
    ]
    
    samples_per_mode = n_samples // len(modes)
    all_samples = []
    
    for mode in modes:
        samples = np.random.multivariate_normal(
            mode, 0.3 * np.eye(2), samples_per_mode
        )
        all_samples.append(samples)
    
    data = np.vstack(all_samples)
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)

def create_two_moons_data(n_samples: int = 1000, noise: float = 0.1) -> torch.Tensor:
    """Create two moons dataset."""
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(X, dtype=torch.float32)

def create_ring_data(n_samples: int = 1000) -> torch.Tensor:
    """Create concentric rings data."""
    # Inner ring
    theta1 = np.random.uniform(0, 2*np.pi, n_samples//2)
    r1 = np.random.normal(1.0, 0.1, n_samples//2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    # Outer ring
    theta2 = np.random.uniform(0, 2*np.pi, n_samples//2)
    r2 = np.random.normal(2.5, 0.1, n_samples//2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    data = np.column_stack([
        np.concatenate([x1, x2]),
        np.concatenate([y1, y2])
    ])
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)

def multimodal_log_prob(x: torch.Tensor) -> torch.Tensor:
    """Log probability for multimodal data."""
    modes = torch.tensor([[-2.0, -1.0], [2.0, 1.0], [0.0, 2.5]], device=x.device)
    
    log_probs = []
    for mode in modes:
        diff = x - mode
        log_prob = -0.5 * (diff**2).sum(dim=1) / 0.3
        log_probs.append(log_prob)
    
    log_probs = torch.stack(log_probs, dim=1)
    return torch.logsumexp(log_probs, dim=1) - np.log(len(modes))

def plot_samples(samples: torch.Tensor, title: str = "Samples", ax=None):
    """Plot 2D samples."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    samples_np = samples.detach().cpu().numpy()
    ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.6, s=20)
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_comparison(target_data: torch.Tensor, model_samples: torch.Tensor, flow_samples: dict):
    """Compare target data with model samples and individual flows."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Target data
    plot_samples(target_data, "Target Data", axes[0, 0])
    
    # Model mixture
    plot_samples(model_samples, "AMF-VI Mixture", axes[0, 1])
    
    # Individual flows
    flow_names = list(flow_samples.keys())
    for i, (name, samples) in enumerate(flow_samples.items()):
        if i < 3:
            row, col = (0, 2) if i == 0 else (1, i-1)
            plot_samples(samples, f"{name.title()} Flow", axes[row, col])
    
    plt.tight_layout()
    return fig