import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from amf_vi.model import SimpleAMFVI
from amf_vi.utils import create_multimodal_data, plot_samples

def compute_coverage(target_samples, generated_samples, threshold=0.1):
    """Compute mode coverage metric."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # For each target sample, find closest generated sample
    distances = cdist(target_np, generated_np)
    min_distances = distances.min(axis=1)
    
    # Coverage: fraction of target samples within threshold
    coverage = (min_distances < threshold).mean()
    return coverage

def compute_quality(target_samples, generated_samples, threshold=0.1):
    """Compute sample quality metric."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # For each generated sample, find closest target sample
    distances = cdist(generated_np, target_np)
    min_distances = distances.min(axis=1)
    
    # Quality: fraction of generated samples within threshold
    quality = (min_distances < threshold).mean()
    return quality

def evaluate_flow_diversity(model, n_samples=500):
    """Evaluate how diverse the flow components are."""
    flow_means = []
    flow_stds = []
    
    for i, flow in enumerate(model.flows):
        samples = flow.sample(n_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        flow_means.append(mean)
        flow_stds.append(std)
    
    # Compute pairwise distances between flow means
    means_tensor = torch.stack(flow_means)
    pairwise_dists = torch.cdist(means_tensor, means_tensor)
    
    # Average distance between different flows (exclude diagonal)
    mask = ~torch.eye(len(flow_means), dtype=bool)
    avg_separation = pairwise_dists[mask].mean()
    
    return {
        'flow_means': flow_means,
        'flow_stds': flow_stds,
        'avg_separation': avg_separation.item(),
        'pairwise_distances': pairwise_dists
    }

def comprehensive_evaluation():
    """Comprehensive evaluation of trained AMF-VI model."""
    
    # Create test data
    test_data = create_multimodal_data(2000)
    
    print("Training AMF-VI model...")
    
    # Create and train model
    model = SimpleAMFVI(dim=2, flow_types=['realnvp', 'planar', 'radial'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Quick training
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(test_data)
        
        # Simple negative log likelihood
        mixture_log_prob = model.log_prob_mixture(test_data)
        loss = -mixture_log_prob.mean() + model.regularization_loss()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("\nEvaluating model...")
    
    # Generate samples for evaluation
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(2000)
        
        # Individual flow samples
        flow_samples = {}
        flow_names = ['realnvp', 'planar', 'radial']
        for i, name in enumerate(flow_names[:len(model.flows)]):
            flow_samples[name] = model.flows[i].sample(1000)
    
    # Compute metrics
    coverage = compute_coverage(test_data, generated_samples)
    quality = compute_quality(test_data, generated_samples)
    diversity_metrics = evaluate_flow_diversity(model)
    
    print(f"\nEvaluation Results:")
    print(f"Coverage: {coverage:.3f}")
    print(f"Quality: {quality:.3f}")
    print(f"Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    
    # Detailed flow analysis
    print(f"\nFlow Component Analysis:")
    for i, (name, samples) in enumerate(flow_samples.items()):
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        print(f"{name.capitalize()} Flow - Mean: [{mean[0]:.2f}, {mean[1]:.2f}], "
              f"Std: [{std[0]:.2f}, {std[1]:.2f}]")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Data and mixture
    plot_samples(test_data, "Target Data", axes[0, 0])
    plot_samples(generated_samples, "AMF-VI Mixture", axes[0, 1])
    
    # Plot flow separation matrix
    pairwise_dists = diversity_metrics['pairwise_distances'].numpy()
    im = axes[0, 2].imshow(pairwise_dists, cmap='viridis')
    axes[0, 2].set_title("Flow Separation Matrix")
    plt.colorbar(im, ax=axes[0, 2])
    
    # Bottom row: Individual flows
    for i, (name, samples) in enumerate(flow_samples.items()):
        if i < 3:
            plot_samples(samples, f"{name.capitalize()} Flow", axes[1, i])
    
    plt.tight_layout()
    plt.show()
    
    return {
        'coverage': coverage,
        'quality': quality,
        'diversity': diversity_metrics,
        'model': model
    }

if __name__ == "__main__":
    results = comprehensive_evaluation()