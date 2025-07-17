import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from amf_vi.model import SimpleAMFVI
from amf_vi.utils import create_multimodal_data, plot_samples
import os
import pickle
import csv
from data.data_generator import generate_data, get_available_datasets

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

def evaluate_single_dataset(dataset_name):
    """Evaluate a single trained model."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    
    # Load pre-trained model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        print("Please run the training script first: python examples/demo_2d_multimodal.py")
        return None
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data['model']
            losses = saved_data.get('losses', [])
        else:
            model = saved_data  # Legacy format
            losses = []
    
    # Ensure model and data are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_data = test_data.to(device)
    
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
    
    # Compute log probability
    with torch.no_grad():
        log_prob = model.log_prob_mixture(test_data).mean().item()
    
    results = {
        'dataset': dataset_name,
        'coverage': coverage,
        'quality': quality,
        'log_probability': log_prob,
        'flow_separation': diversity_metrics['avg_separation'],
        'diversity_metrics': diversity_metrics,
        'flow_samples': flow_samples,
        'generated_samples': generated_samples,
        'test_data': test_data,
        'model': model,
        'losses': losses
    }
    
    print(f"📊 Results for {dataset_name}:")
    print(f"   Coverage: {coverage:.3f}")
    print(f"   Quality: {quality:.3f}")
    print(f"   Log Probability: {log_prob:.3f}")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    
    # Detailed flow analysis
    print(f"   Flow Components:")
    for i, (name, samples) in enumerate(flow_samples.items()):
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        print(f"     {name.capitalize()} - Mean: [{mean[0]:.2f}, {mean[1]:.2f}], "
              f"Std: [{std[0]:.2f}, {std[1]:.2f}]")
    
    return results

def comprehensive_evaluation():
    """Comprehensive evaluation of all trained AMF-VI models."""
    
    # Define datasets to evaluate
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        results = evaluate_single_dataset(dataset_name)
        if results is not None:
            all_results[dataset_name] = results
    
    if not all_results:
        print("❌ No trained models found. Please run training first.")
        return None
    
    # Create comprehensive visualization
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print(f"{'='*60}")
    
    n_datasets = len(all_results)
    fig, axes = plt.subplots(3, n_datasets, figsize=(5*n_datasets, 12))
    
    # Make axes 2D for consistent indexing
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for different datasets
    colors = ['steelblue', 'crimson', 'forestgreen', 'darkorange']
    
    for i, (dataset_name, results) in enumerate(all_results.items()):
        # Row 1: Original data
        plot_samples(results['test_data'], f"{dataset_name.title()} Data", 
                    axes[0, i], color=colors[i % len(colors)])
        
        # Row 2: Generated samples
        plot_samples(results['generated_samples'], f"Generated Samples", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Flow separation matrix
        pairwise_dists = results['diversity_metrics']['pairwise_distances'].detach().cpu().numpy()
        im = axes[2, i].imshow(pairwise_dists, cmap='viridis')
        axes[2, i].set_title(f"Flow Separation")
        plt.colorbar(im, ax=axes[2, i])
    
    plt.tight_layout()
    plt.suptitle('AMF-VI Evaluation Results - All Datasets', fontsize=16, y=0.98)
    
    # Save comprehensive plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Flow Sep':<8}")
    print("-" * 80)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<15} | {results['coverage']:<8.3f} | {results['quality']:<8.3f} | "
              f"{results['log_probability']:<10.3f} | {results['flow_separation']:<8.3f}")
        
        summary_data.append([
            dataset_name,
            results['coverage'],
            results['quality'],
            results['log_probability'],
            results['flow_separation']
        ])
    
    # Save comprehensive results
    
    # Save summary to CSV
    with open(os.path.join(results_dir, 'comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'coverage', 'quality', 'log_probability', 'flow_separation'])
        writer.writerows(summary_data)
    
    # Save detailed results to text file
    with open(os.path.join(results_dir, 'comprehensive_evaluation.txt'), 'w') as f:
        f.write("Comprehensive Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for dataset_name, results in all_results.items():
            f.write(f"Dataset: {dataset_name.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Coverage: {results['coverage']:.6f}\n")
            f.write(f"Quality: {results['quality']:.6f}\n")
            f.write(f"Log Probability: {results['log_probability']:.6f}\n")
            f.write(f"Flow Separation: {results['flow_separation']:.6f}\n")
            f.write("\nFlow Analysis:\n")
            for name, samples in results['flow_samples'].items():
                mean = samples.mean(dim=0)
                std = samples.std(dim=0)
                f.write(f"  {name.capitalize()} Flow - Mean: [{mean[0]:.2f}, {mean[1]:.2f}], "
                       f"Std: [{std[0]:.2f}, {std[1]:.2f}]\n")
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, 'comprehensive_evaluation.txt')}")
    print(f"   Metrics saved to: {os.path.join(results_dir, 'comprehensive_metrics.csv')}")
    print(f"   Plots saved to: {os.path.join(results_dir, 'comprehensive_evaluation.png')}")
    
    # Best performing dataset
    best_dataset = max(all_results.items(), key=lambda x: x[1]['coverage'])
    print(f"\n🏆 Best performing dataset: {best_dataset[0]} (Coverage: {best_dataset[1]['coverage']:.3f})")
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_evaluation()