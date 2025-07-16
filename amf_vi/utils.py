import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from typing import Tuple, Callable, List

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

def plot_samples(samples: torch.Tensor, title: str = "Samples", ax=None, color=None, alpha=0.6, s=20):
    """Plot 2D samples."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    samples_np = samples.detach().cpu().numpy()
    scatter = ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=alpha, s=s, c=color)
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(True, alpha=0.3)
    
    return ax, scatter

def plot_comparison(target_data: torch.Tensor, model_samples: torch.Tensor, flow_samples: dict):
    """Compare target data with model samples and individual flows."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Target data
    plot_samples(target_data, "Target Data", axes[0, 0], color='blue')
    
    # Model mixture
    plot_samples(model_samples, "AMF-VI Mixture", axes[0, 1], color='red')
    
    # Individual flows
    flow_names = list(flow_samples.keys())
    colors = ['green', 'orange', 'purple']
    
    for i, (name, samples) in enumerate(flow_samples.items()):
        if i < 3:
            row, col = (0, 2) if i == 0 else (1, i-1)
            color = colors[i] if i < len(colors) else None
            plot_samples(samples, f"{name.title()} Flow", axes[row, col], color=color)
    
    plt.tight_layout()
    return fig

def plot_training_progress(losses: List[float], title: str = "Training Loss"):
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(losses) > 10:
        z = np.polyfit(range(len(losses)), losses, 1)
        p = np.poly1d(z)
        ax.plot(range(len(losses)), p(range(len(losses))), "--", alpha=0.8, color='red', label='Trend')
        ax.legend()
    
    return fig

def plot_density_heatmap(model, xlim=(-4, 4), ylim=(-3, 4), resolution=100):
    """Plot density heatmap of the learned distribution."""
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create grid points
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)
    
    # Compute log probabilities
    model.eval()
    with torch.no_grad():
        log_probs = model.log_prob_mixture(grid_points)
        probs = torch.exp(log_probs).numpy().reshape(X.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, probs, levels=20, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, probs, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    plt.colorbar(contour, ax=ax, label='Probability Density')
    ax.set_title('Learned Probability Density')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_flow_weights_distribution(model, data_sample: torch.Tensor):
    """Visualize how the gating network assigns weights to different flows."""
    model.eval()
    with torch.no_grad():
        weights = model.gating_net(data_sample)  # [batch, n_flows]
    
    weights_np = weights.cpu().numpy()
    flow_names = ['Real-NVP', 'Planar', 'Radial'][:weights.shape[1]]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of weights for each flow
    for i, name in enumerate(flow_names):
        axes[0].hist(weights_np[:, i], alpha=0.6, label=name, bins=30)
    axes[0].set_title('Distribution of Flow Weights')
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Average weights
    avg_weights = weights_np.mean(axis=0)
    axes[1].bar(flow_names, avg_weights, color=['blue', 'orange', 'green'][:len(flow_names)])
    axes[1].set_title('Average Flow Weights')
    axes[1].set_ylabel('Average Weight')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_comprehensive_results(target_data: torch.Tensor, model, losses: List[float] = None):
    """Create a comprehensive visualization of all results."""
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000)
        
        # Individual flow samples
        flow_samples = {}
        flow_names = ['realnvp', 'planar', 'radial']
        for i, name in enumerate(flow_names[:len(model.flows)]):
            flow_samples[name] = model.flows[i].sample(1000)
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Main comparison (top row)
        ax1 = plt.subplot(3, 4, 1)
        plot_samples(target_data, "Target Data", ax1, color='blue')
        
        ax2 = plt.subplot(3, 4, 2)
        plot_samples(model_samples, "AMF-VI Mixture", ax2, color='red')
        
        # 2. Individual flows
        colors = ['green', 'orange', 'purple']
        for i, (name, samples) in enumerate(flow_samples.items()):
            if i < 2:  # First two flows in top row
                ax = plt.subplot(3, 4, 3 + i)
                plot_samples(samples, f"{name.title()} Flow", ax, color=colors[i])
        
        # 3. Density heatmap
        ax5 = plt.subplot(3, 4, 5)
        try:
            x = np.linspace(-4, 4, 50)
            y = np.linspace(-3, 4, 50)
            X, Y = np.meshgrid(x, y)
            grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)
            log_probs = model.log_prob_mixture(grid_points)
            probs = torch.exp(log_probs).numpy().reshape(X.shape)
            
            contour = ax5.contourf(X, Y, probs, levels=15, cmap='viridis', alpha=0.8)
            ax5.set_title('Learned Density')
            ax5.set_xlabel('X1')
            ax5.set_ylabel('X2')
        except:
            ax5.text(0.5, 0.5, 'Density plot\nunavailable', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Learned Density')
        
        # 4. Training loss (if available)
        if losses:
            ax6 = plt.subplot(3, 4, 6)
            ax6.plot(losses, linewidth=2)
            ax6.set_title('Training Loss')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss')
            ax6.grid(True, alpha=0.3)
        
        # 5. Flow weights distribution
        ax7 = plt.subplot(3, 4, 7)
        weights = model.gating_net(target_data[:500])
        weights_np = weights.cpu().numpy()
        avg_weights = weights_np.mean(axis=0)
        flow_labels = ['Real-NVP', 'Planar', 'Radial'][:len(avg_weights)]
        ax7.bar(flow_labels, avg_weights, color=colors[:len(avg_weights)])
        ax7.set_title('Average Flow Weights')
        ax7.set_ylabel('Weight')
        
        # 6. Statistics comparison
        ax8 = plt.subplot(3, 4, 8)
        target_mean = target_data.mean(dim=0).cpu().numpy()
        model_mean = model_samples.mean(dim=0).cpu().numpy()
        target_std = target_data.std(dim=0).cpu().numpy()
        model_std = model_samples.std(dim=0).cpu().numpy()
        
        x_pos = np.arange(2)
        width = 0.35
        
        ax8.bar(x_pos - width/2, [target_mean[0], target_mean[1]], width, label='Target Mean', alpha=0.7)
        ax8.bar(x_pos + width/2, [model_mean[0], model_mean[1]], width, label='Model Mean', alpha=0.7)
        ax8.set_title('Mean Comparison')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(['X1', 'X2'])
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 7-9. Individual flow samples (bottom row)
        for i, (name, samples) in enumerate(flow_samples.items()):
            if i < 3:
                ax = plt.subplot(3, 4, 9 + i)
                plot_samples(samples, f"{name.title()}", ax, color=colors[i])
        
        # 10. Overlay comparison
        ax12 = plt.subplot(3, 4, 12)
        plot_samples(target_data, "Overlay Comparison", ax12, color='blue', alpha=0.4, s=15)
        plot_samples(model_samples, "", ax12, color='red', alpha=0.4, s=15)
        ax12.legend(['Target', 'Generated'], loc='upper right')
        
        plt.tight_layout()
        return fig

def save_all_plots(target_data: torch.Tensor, model, losses: List[float] = None, save_dir: str = "./results"):
    """Save all visualization plots to files."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Comprehensive results
    fig1 = plot_comprehensive_results(target_data, model, losses)
    fig1.savefig(f"{save_dir}/comprehensive_results.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Individual comparison
    with torch.no_grad():
        model_samples = model.sample(1000)
        flow_samples = {}
        flow_names = ['realnvp', 'planar', 'radial']
        for i, name in enumerate(flow_names[:len(model.flows)]):
            flow_samples[name] = model.flows[i].sample(1000)
    
    fig2 = plot_comparison(target_data, model_samples, flow_samples)
    fig2.savefig(f"{save_dir}/flow_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Training progress
    if losses:
        fig3 = plot_training_progress(losses)
        fig3.savefig(f"{save_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    # 4. Density heatmap
    try:
        fig4 = plot_density_heatmap(model)
        fig4.savefig(f"{save_dir}/density_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
    except:
        print("Could not generate density heatmap")
    
    # 5. Flow weights
    fig5 = plot_flow_weights_distribution(model, target_data[:500])
    fig5.savefig(f"{save_dir}/flow_weights.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    print(f"All plots saved to {save_dir}/")