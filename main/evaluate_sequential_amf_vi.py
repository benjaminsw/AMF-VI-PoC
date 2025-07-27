import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from wgf_amf_vi import WassersteinGradientFlowAMFVI, train_wgf_amf_vi
from data.data_generator import generate_data
import os
import pickle
import csv

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

def evaluate_individual_flows(model, test_data, flow_names):
    """Evaluate each individual flow against test data."""
    individual_metrics = {}
    
    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            # Generate samples from individual flow
            flow_samples = flow.sample(2000)
            
            # Compute metrics for this flow
            coverage = compute_coverage(test_data, flow_samples)
            quality = compute_quality(test_data, flow_samples)
            log_prob = flow.log_prob(test_data).mean().item()
            
            # Store metrics
            individual_metrics[name] = {
                'coverage': coverage,
                'quality': quality,
                'log_probability': log_prob,
                'samples': flow_samples
            }
    
    return individual_metrics

def evaluate_single_wgf_dataset(dataset_name):
    """Evaluate or train+evaluate a single WGF model."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating WGF {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_data.to(device)
    
    # Check if model exists, if not train it
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'wgf_model_{dataset_name}.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing WGF model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            losses = saved_data.get('losses', [])
    else:
        print(f"Training new WGF model for {dataset_name}")
        model, flow_losses, wgf_losses = train_wgf_amf_vi(dataset_name, show_plots=False, save_plots=False)
        
        # Save the trained model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'flow_losses': flow_losses,
                'wgf_losses': wgf_losses,
                'dataset': dataset_name
            }, f)
        losses = flow_losses
    
    model = model.to(device)
    flow_names = ['realnvp', 'maf', 'iaf']
    
    # Generate samples for evaluation
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(2000)
        
        # Individual flow samples
        flow_samples = {}
        for i, name in enumerate(flow_names[:len(model.flows)]):
            flow_samples[name] = model.flows[i].sample(1000)
    
    # Compute overall mixture metrics
    coverage = compute_coverage(test_data, generated_samples)
    quality = compute_quality(test_data, generated_samples)
    diversity_metrics = evaluate_flow_diversity(model)
    
    # For WGF model, use the forward method to get log probability
    with torch.no_grad():
        result = model.forward(test_data)
        log_prob = result['mixture_log_prob'].mean().item()
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names[:len(model.flows)])
    
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
        'losses': losses,
        'individual_flow_metrics': individual_flow_metrics,
        'final_weights': model.flow_weights.detach().cpu().numpy()
    }
    
    print(f"📊 Overall WGF Mixture Results for {dataset_name}:")
    print(f"   Coverage: {coverage:.3f}")
    print(f"   Quality: {quality:.3f}")
    print(f"   Log Probability: {log_prob:.3f}")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    print(f"   Final Weights: {results['final_weights']}")
    
    # Print individual flow performance
    print(f"\n📊 Individual Flow Results:")
    for name, metrics in individual_flow_metrics.items():
        print(f"   {name.upper()} Flow:")
        print(f"     Coverage: {metrics['coverage']:.3f}")
        print(f"     Quality: {metrics['quality']:.3f}")
        print(f"     Log Probability: {metrics['log_probability']:.3f}")
    
    return results

def comprehensive_wgf_evaluation():
    """Comprehensive evaluation of all WGF AMF-VI models."""
    
    # Define datasets to evaluate
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        results = evaluate_single_wgf_dataset(dataset_name)
        if results is not None:
            all_results[dataset_name] = results
    
    if not all_results:
        print("❌ No WGF models could be trained/evaluated.")
        return None
    
    # Create comprehensive visualization
    print(f"\n{'='*60}")
    print("CREATING WGF COMPREHENSIVE VISUALIZATION")
    print(f"{'='*60}")
    
    n_datasets = len(all_results)
    fig, axes = plt.subplots(3, n_datasets, figsize=(5*n_datasets, 12))
    
    # Make axes 2D for consistent indexing
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for different datasets
    colors = ['steelblue', 'crimson', 'forestgreen', 'darkorange']
    
    def plot_samples(samples, title, ax, color, alpha=0.6, s=20):
        """Plot 2D samples."""
        samples_np = samples.detach().cpu().numpy()
        ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=alpha, s=s, c=color)
        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.grid(True, alpha=0.3)
    
    for i, (dataset_name, results) in enumerate(all_results.items()):
        # Row 1: Original data
        plot_samples(results['test_data'], f"{dataset_name.title()} Data", 
                    axes[0, i], color=colors[i % len(colors)])
        
        # Row 2: Generated samples
        plot_samples(results['generated_samples'], f"WGF Generated Samples", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Flow separation matrix
        pairwise_dists = results['diversity_metrics']['pairwise_distances'].detach().cpu().numpy()
        im = axes[2, i].imshow(pairwise_dists, cmap='viridis')
        axes[2, i].set_title(f"Flow Separation")
        plt.colorbar(im, ax=axes[2, i])
    
    plt.tight_layout()
    plt.suptitle('WGF AMF-VI Evaluation Results - All Datasets', fontsize=16, y=0.98)
    
    # Save comprehensive plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'wgf_comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create summary table
    print(f"\n{'='*80}")
    print("WGF SUMMARY TABLE - MIXTURE RESULTS")
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
    
    # Save mixture metrics to CSV
    with open(os.path.join(results_dir, 'wgf_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'coverage', 'quality', 'log_probability', 'flow_separation'])
        writer.writerows(summary_data)
    
    # Create individual flow metrics CSV
    individual_flow_data = []
    
    print(f"\n{'='*80}")
    print("WGF INDIVIDUAL FLOW COMPARISON")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"{'Flow':<8} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10}")
        print("-" * 50)
        
        if 'individual_flow_metrics' in results:
            for flow_name, metrics in results['individual_flow_metrics'].items():
                print(f"{flow_name.upper():<8} | {metrics['coverage']:<8.3f} | {metrics['quality']:<8.3f} | "
                      f"{metrics['log_probability']:<10.3f}")
                
                individual_flow_data.append([
                    dataset_name,
                    flow_name,
                    metrics['coverage'],
                    metrics['quality'],
                    metrics['log_probability']
                ])
    
    # Save individual flow metrics to CSV
    with open(os.path.join(results_dir, 'wgf_individual_flow_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow_name', 'coverage', 'quality', 'log_probability'])
        writer.writerows(individual_flow_data)
    
    print(f"\n✅ WGF Comprehensive evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, 'wgf_comprehensive_evaluation.png')}")
    print(f"   Mixture metrics saved to: {os.path.join(results_dir, 'wgf_comprehensive_metrics.csv')}")
    print(f"   Individual flow metrics saved to: {os.path.join(results_dir, 'wgf_individual_flow_metrics.csv')}")
    
    # Best performing dataset (mixture)
    best_dataset = max(all_results.items(), key=lambda x: x[1]['coverage'])
    print(f"\n🏆 Best performing WGF dataset (mixture): {best_dataset[0]} (Coverage: {best_dataset[1]['coverage']:.3f})")
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_wgf_evaluation()
