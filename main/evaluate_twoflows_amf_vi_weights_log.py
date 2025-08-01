import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from twoflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
from data.data_generator import generate_data
import os
import pickle
import csv

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

def get_target_log_prob(dataset_name, samples):
    """Get target log probabilities for known datasets."""
    # This is a placeholder - you'll need to implement the actual target log prob
    # For now, we'll use a simple approach based on the dataset type
    if dataset_name in ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']:
        # For these datasets, we can approximate with a simple Gaussian mixture
        # This is a simplification - ideally you'd have the exact target log prob
        return torch.zeros(samples.shape[0])  # Placeholder
    else:
        return torch.zeros(samples.shape[0])  # Placeholder

def compute_cross_entropy_surrogate(target_samples, flow_model):
    """Compute cross-entropy surrogate for KL divergence: -E_p[log q(x)]"""
    with torch.no_grad():
        log_q = flow_model.log_prob(target_samples)
        return -log_q.mean().item()

def compute_percentage_improvement(target_samples, mixture_model, baseline_flow):
    """Compute percentage improvement of mixture model over a single baseline flow."""
    mixture_cross_entropy = compute_cross_entropy_surrogate(target_samples, mixture_model)
    baseline_cross_entropy = compute_cross_entropy_surrogate(target_samples, baseline_flow)
    
    if baseline_cross_entropy == 0:
        return 0.0
    
    improvement = ((baseline_cross_entropy - mixture_cross_entropy) / baseline_cross_entropy) * 100
    return improvement

def compute_kl_divergence_metric(target_samples, flow_model, dataset_name):
    """Compute KL divergence using direct mathematical approach."""
    with torch.no_grad():
        # Get flow log probabilities
        log_q = flow_model.log_prob(target_samples)
        
        # Get target log probabilities (this needs to be implemented for each dataset)
        log_p = get_target_log_prob(dataset_name, target_samples)
        
        # For now, fall back to histogram method since we don't have exact target log probs
        # TODO: Implement exact target log probabilities for each dataset
        return compute_kl_divergence_histogram(target_samples, flow_model.sample(2000))

def compute_kl_divergence_histogram(target_samples, generated_samples):
    """Compute KL divergence between target and generated samples using histogram method."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # Simple histogram-based KL divergence estimation
    bins = 50
    
    # Get data range
    x_min = min(target_np[:, 0].min(), generated_np[:, 0].min())
    x_max = max(target_np[:, 0].max(), generated_np[:, 0].max())
    y_min = min(target_np[:, 1].min(), generated_np[:, 1].min())
    y_max = max(target_np[:, 1].max(), generated_np[:, 1].max())
    
    # Create histograms
    hist_target, _, _ = np.histogram2d(target_np[:, 0], target_np[:, 1], 
                                       bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    hist_generated, _, _ = np.histogram2d(generated_np[:, 0], generated_np[:, 1], 
                                          bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    
    # Normalize to probabilities
    hist_target = hist_target / hist_target.sum()
    hist_generated = hist_generated / hist_generated.sum()
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_target = hist_target + epsilon
    hist_generated = hist_generated + epsilon
    
    # Compute KL divergence
    kl_div = np.sum(hist_target * np.log(hist_target / hist_generated))
    
    return kl_div

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

def evaluate_individual_flows(model, test_data, flow_names, dataset_name):
    """Evaluate each individual flow against test data."""
    individual_metrics = {}
    
    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            # Generate samples from individual flow
            flow_samples = flow.sample(2000)
            
            # Compute metrics: KL divergence and cross-entropy surrogate only
            kl_divergence = compute_kl_divergence_metric(test_data, flow, dataset_name)
            cross_entropy = compute_cross_entropy_surrogate(test_data, flow)
            
            # Store metrics
            individual_metrics[name] = {
                'kl_divergence': kl_divergence,
                'cross_entropy_surrogate': cross_entropy,
                'samples': flow_samples
            }
    
    return individual_metrics

def evaluate_single_sequential_dataset(dataset_name):
    """Evaluate or train+evaluate a single Sequential model."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating Sequential {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_data.to(device)
    
    # Check if model exists, if not train it
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing Sequential model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            flow_losses = saved_data.get('flow_losses', [])
            weight_losses = saved_data.get('weight_losses', [])
    else:
        print(f"Training new Sequential model for {dataset_name}")
        model, flow_losses, weight_losses = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)
        
        # Save the trained model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'flow_losses': flow_losses,
                'weight_losses': weight_losses,
                'dataset': dataset_name
            }, f)
    
    model = model.to(device)
    flow_names = ['realnvp', 'maf']
    
    # Generate samples for evaluation
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(2000)
        
        # Individual flow samples
        flow_samples = {}
        for i, name in enumerate(flow_names[:len(model.flows)]):
            flow_samples[name] = model.flows[i].sample(1000)
    
    # Compute metrics: KL divergence, cross-entropy surrogate, and percentage improvements
    kl_divergence = compute_kl_divergence_metric(test_data, model, dataset_name)
    cross_entropy = compute_cross_entropy_surrogate(test_data, model)
    percentage_improvement_vs_realnvp = compute_percentage_improvement(test_data, model, model.flows[0])
    percentage_improvement_vs_maf = compute_percentage_improvement(test_data, model, model.flows[1])
    diversity_metrics = evaluate_flow_diversity(model)
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names[:len(model.flows)], dataset_name)
    
    # Get learned weights (handle both log_weights and weights attributes)
    if model.weights_trained:
        print('*** learned weights is extracted ***')
        if hasattr(model, 'log_weights'):
            learned_weights = F.softmax(model.log_weights, dim=0).detach().cpu().numpy()
        else:
            learned_weights = model.weights.detach().cpu().numpy()
    else:
        learned_weights = np.ones(len(model.flows)) / len(model.flows)
    
    results = {
        'dataset': dataset_name,
        'kl_divergence': kl_divergence,
        'cross_entropy_surrogate': cross_entropy,
        'percentage_improvement_vs_realnvp': percentage_improvement_vs_realnvp,
        'percentage_improvement_vs_maf': percentage_improvement_vs_maf,
        'flow_separation': diversity_metrics['avg_separation'],
        'diversity_metrics': diversity_metrics,
        'flow_samples': flow_samples,
        'generated_samples': generated_samples,
        'test_data': test_data,
        'model': model,
        'flow_losses': flow_losses,
        'weight_losses': weight_losses,
        'individual_flow_metrics': individual_flow_metrics,
        'learned_weights': learned_weights,
        'weights_trained': model.weights_trained
    }
    
    print(f"📊 Overall Sequential Mixture Results for {dataset_name}:")
    print(f"   KL Divergence: {kl_divergence:.3f}")
    print(f"   Cross-Entropy Surrogate: {cross_entropy:.3f}")
    print(f"   % Improvement vs RealNVP: {percentage_improvement_vs_realnvp:.1f}%")
    print(f"   % Improvement vs MAF: {percentage_improvement_vs_maf:.1f}%")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    print(f"   Learned Weights: {learned_weights}")
    print(f"   Weights Trained: {model.weights_trained}")
    
    # Print individual flow performance
    print(f"\n📊 Individual Flow Results:")
    for name, metrics in individual_flow_metrics.items():
        print(f"   {name.upper()} Flow:")
        print(f"     KL Divergence: {metrics['kl_divergence']:.3f}")
        print(f"     Cross-Entropy Surrogate: {metrics['cross_entropy_surrogate']:.3f}")
    
    return results

def comprehensive_sequential_evaluation():
    """Comprehensive evaluation of all Sequential AMF-VI models."""
    
    # Define datasets to evaluate (including new ones)
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        try:
            results = evaluate_single_sequential_dataset(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"❌ Failed to evaluate {dataset_name}: {e}")
            continue
    
    if not all_results:
        print("❌ No Sequential models could be trained/evaluated.")
        return None
    
    # Create comprehensive visualization
    print(f"\n{'='*60}")
    print("CREATING SEQUENTIAL COMPREHENSIVE VISUALIZATION")
    print(f"{'='*60}")
    
    n_datasets = len(all_results)
    fig, axes = plt.subplots(3, n_datasets, figsize=(5*n_datasets, 12))
    
    # Make axes 2D for consistent indexing
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for different datasets
    colors = ['steelblue', 'crimson', 'forestgreen', 'darkorange', 'purple', 'brown', 'pink']
    
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
        plot_samples(results['generated_samples'], f"Sequential Generated Samples", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Weight visualization or flow separation
        if results['weights_trained']:
            # Plot learned weights
            weights = results['learned_weights']
            flow_names = ['RealNVP', 'MAF'][:len(weights)]
            bars = axes[2, i].bar(flow_names, weights, color=['green', 'orange'][:len(weights)])
            axes[2, i].set_title(f"Learned Weights")
            axes[2, i].set_ylabel('Weight')
            axes[2, i].set_ylim(0, 1)
            # Add weight values on bars
            for bar, weight in zip(bars, weights):
                axes[2, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{weight:.3f}', ha='center', va='bottom')
        else:
            # Fallback to flow separation matrix
            pairwise_dists = results['diversity_metrics']['pairwise_distances'].detach().cpu().numpy()
            im = axes[2, i].imshow(pairwise_dists, cmap='viridis')
            axes[2, i].set_title(f"Flow Separation")
            plt.colorbar(im, ax=axes[2, i])
    
    plt.tight_layout()
    plt.suptitle('Sequential AMF-VI Evaluation Results - All Datasets', fontsize=16, y=0.98)
    
    # Save comprehensive plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'sequential_comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create summary table with new metrics
    print(f"\n{'='*80}")
    print("SEQUENTIAL SUMMARY TABLE - MIXTURE RESULTS")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Flow':<8} | {'KL Divergence':<12} | {'Cross-Entropy':<12} | {'% Improvement':<12} | {'Flow Sep':<8} | {'Weights Trained':<14}")
    print("-" * 110)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"
        
        # Add row for RealNVP comparison
        print(f"{dataset_name:<15} | {'RealNVP':<8} | {results['kl_divergence']:<12.3f} | "
              f"{results['cross_entropy_surrogate']:<12.3f} | {results['percentage_improvement_vs_realnvp']:<12.1f} | "
              f"{results['flow_separation']:<8.3f} | {weights_status:<14}")
        
        summary_data.append([
            dataset_name,
            'RealNVP',
            results['kl_divergence'],
            results['cross_entropy_surrogate'],
            results['percentage_improvement_vs_realnvp'],
            results['flow_separation'],
            weights_status
        ])
        
        # Add row for MAF comparison
        print(f"{dataset_name:<15} | {'MAF':<8} | {results['kl_divergence']:<12.3f} | "
              f"{results['cross_entropy_surrogate']:<12.3f} | {results['percentage_improvement_vs_maf']:<12.1f} | "
              f"{results['flow_separation']:<8.3f} | {weights_status:<14}")
        
        summary_data.append([
            dataset_name,
            'MAF',
            results['kl_divergence'],
            results['cross_entropy_surrogate'],
            results['percentage_improvement_vs_maf'],
            results['flow_separation'],
            weights_status
        ])
    
    # Save mixture metrics to CSV with new metrics
    with open(os.path.join(results_dir, 'sequential_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow', 'kl_divergence', 'cross_entropy_surrogate', 'percentage_improvement', 'flow_separation', 'weights_trained'])
        writer.writerows(summary_data)
    
    # Create individual flow metrics CSV with new metrics
    individual_flow_data = []
    
    print(f"\n{'='*80}")
    print("SEQUENTIAL INDIVIDUAL FLOW COMPARISON")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"{'Flow':<8} | {'KL Divergence':<12} | {'Cross-Entropy':<12} | {'Weight':<8}")
        print("-" * 55)
        
        if 'individual_flow_metrics' in results:
            for i, (flow_name, metrics) in enumerate(results['individual_flow_metrics'].items()):
                weight = results['learned_weights'][i] if results['weights_trained'] else 0.5
                print(f"{flow_name.upper():<8} | {metrics['kl_divergence']:<12.3f} | "
                      f"{metrics['cross_entropy_surrogate']:<12.3f} | {weight:<8.3f}")
                
                individual_flow_data.append([
                    dataset_name,
                    flow_name,
                    metrics['kl_divergence'],
                    metrics['cross_entropy_surrogate'],
                    weight
                ])
    
    # Save individual flow metrics to CSV with new metrics
    with open(os.path.join(results_dir, 'sequential_individual_flow_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow_name', 'kl_divergence', 'cross_entropy_surrogate', 'learned_weight'])
        writer.writerows(individual_flow_data)
    
    print(f"\n✅ Sequential Comprehensive evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, 'sequential_comprehensive_evaluation.png')}")
    print(f"   Mixture metrics saved to: {os.path.join(results_dir, 'sequential_comprehensive_metrics.csv')}")
    print(f"   Individual flow metrics saved to: {os.path.join(results_dir, 'sequential_individual_flow_metrics.csv')}")
    
    # Best performing dataset based on KL divergence (lower is better)
    best_kl_dataset = min(all_results.items(), key=lambda x: x[1]['kl_divergence'])
    print(f"\n🏆 Best performing Sequential dataset (KL divergence): {best_kl_dataset[0]} (KL Divergence: {best_kl_dataset[1]['kl_divergence']:.3f})")
    
    # Best performing dataset based on percentage improvement vs RealNVP (higher is better)
    realnvp_data = [(name, results['percentage_improvement_vs_realnvp']) for name, results in all_results.items()]
    best_improvement_realnvp = max(realnvp_data, key=lambda x: x[1])
    print(f"🏆 Best performing Sequential dataset (% improvement vs RealNVP): {best_improvement_realnvp[0]} (Improvement: {best_improvement_realnvp[1]:.1f}%)")
    
    # Best performing dataset based on percentage improvement vs MAF (higher is better)
    maf_data = [(name, results['percentage_improvement_vs_maf']) for name, results in all_results.items()]
    best_improvement_maf = max(maf_data, key=lambda x: x[1])
    print(f"🏆 Best performing Sequential dataset (% improvement vs MAF): {best_improvement_maf[0]} (Improvement: {best_improvement_maf[1]:.1f}%)")
    
    # Weight learning analysis
    print(f"\n📊 Weight Learning Analysis:")
    for dataset_name, results in all_results.items():
        if results['weights_trained']:
            weights = results['learned_weights']
            entropy = -np.sum(weights * np.log(weights + 1e-8))  # Weight entropy
            print(f"   {dataset_name}: Weights={weights}, Entropy={entropy:.3f}")
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_sequential_evaluation()