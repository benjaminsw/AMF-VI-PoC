import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sequential_amf_vi_adaptive import SequentialAMFVI
from data.data_generator import generate_data, get_available_datasets
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

def evaluate_meta_learner_weights(model, test_data, flow_names):
    """Evaluate meta-learner weight distribution and specialization."""
    if not model.meta_trained:
        return None
    
    with torch.no_grad():
        # Get adaptive weights for test data
        model_output = model.forward(test_data)
        weights = model_output['weights'].cpu().numpy()
        
        # Analyze weight statistics
        weight_stats = {}
        for i, name in enumerate(flow_names):
            weight_stats[name] = {
                'mean': weights[:, i].mean(),
                'std': weights[:, i].std(),
                'min': weights[:, i].min(),
                'max': weights[:, i].max(),
                'entropy': -np.mean(weights[:, i] * np.log(weights[:, i] + 1e-8))
            }
        
        # Overall weight diversity (entropy)
        weight_entropy = -np.mean(np.sum(weights * np.log(weights + 1e-8), axis=1))
        
        return {
            'weight_stats': weight_stats,
            'weight_entropy': weight_entropy,
            'weights_matrix': weights
        }

def evaluate_adaptive_sampling_behavior(model, n_samples=1000):
    """NEW: Evaluate adaptive sampling behavior and effectiveness."""
    if not model.meta_trained:
        return None
    
    # Generate samples using adaptive sampling
    print("🎯 Evaluating adaptive sampling behavior...")
    adaptive_samples = model.sample(n_samples)
    
    # Generate samples using uniform sampling for comparison
    print("📊 Generating uniform samples for comparison...")
    uniform_samples = model._uniform_sample(n_samples)
    
    # Compute sample statistics
    adaptive_stats = {
        'mean': adaptive_samples.mean(dim=0).cpu().numpy(),
        'std': adaptive_samples.std(dim=0).cpu().numpy(),
        'samples': adaptive_samples
    }
    
    uniform_stats = {
        'mean': uniform_samples.mean(dim=0).cpu().numpy(),
        'std': uniform_samples.std(dim=0).cpu().numpy(), 
        'samples': uniform_samples
    }
    
    return {
        'adaptive_stats': adaptive_stats,
        'uniform_stats': uniform_stats
    }

# CHANGE: Add direct flow performance evaluation
def evaluate_direct_vs_adaptive_performance(model, test_data):
    """Compare direct flow performance vs adaptive mixture performance."""
    if not model.flows_trained:
        return None
    
    with torch.no_grad():
        # Get direct flow performance (without meta-learner) - use manual calculation
        if len(model.flows) == 1:
            # Single flow case
            direct_log_prob = model.flows[0].log_prob(test_data).mean()
        else:
            # Multiple flows - use uniform mixture
            flow_log_probs = []
            for flow in model.flows:
                log_prob = flow.log_prob(test_data)
                flow_log_probs.append(log_prob.unsqueeze(1))
            
            flow_predictions = torch.cat(flow_log_probs, dim=1)
            uniform_weights = torch.ones_like(flow_predictions) / len(model.flows)
            weighted_log_probs = flow_predictions + torch.log(uniform_weights + 1e-8)
            direct_log_prob = torch.logsumexp(weighted_log_probs, dim=1).mean()
        
        # Generate direct flow samples for coverage/quality comparison
        if len(model.flows) == 1:
            # Single flow case
            direct_samples = model.flows[0].sample(1000)
            direct_coverage = compute_coverage(test_data, direct_samples)
            direct_quality = compute_quality(test_data, direct_samples)
        else:
            # Multiple flows - use uniform mixture sampling
            direct_samples = model._uniform_sample(1000)
            direct_coverage = compute_coverage(test_data, direct_samples)
            direct_quality = compute_quality(test_data, direct_samples)
        
        # Get adaptive performance if meta-learner is trained
        if model.meta_trained:
            adaptive_output = model.forward(test_data)
            adaptive_log_prob = adaptive_output['mixture_log_prob'].mean().item()
            
            # Generate adaptive samples
            adaptive_samples = model.sample(1000)
            adaptive_coverage = compute_coverage(test_data, adaptive_samples)
            adaptive_quality = compute_quality(test_data, adaptive_samples)
            
            return {
                'direct_log_prob': direct_log_prob.item(),
                'adaptive_log_prob': adaptive_log_prob,
                'direct_coverage': direct_coverage,
                'adaptive_coverage': adaptive_coverage,
                'direct_quality': direct_quality,
                'adaptive_quality': adaptive_quality,
                'log_prob_improvement': adaptive_log_prob - direct_log_prob.item(),
                'coverage_improvement': adaptive_coverage - direct_coverage,
                'quality_improvement': adaptive_quality - direct_quality,
                'direct_samples': direct_samples,
                'adaptive_samples': adaptive_samples
            }
        else:
            return {
                'direct_log_prob': direct_log_prob.item(),
                'direct_coverage': direct_coverage,
                'direct_quality': direct_quality,
                'direct_samples': direct_samples
            }

def evaluate_single_dataset(dataset_name):
    """Evaluate a single trained sequential model with adaptive sampling."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name.upper()} dataset (Adaptive Sampling)")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    
    # CHANGE: Try different model file patterns for single flow testing
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Try different model file naming patterns
    possible_paths = [
        os.path.join(results_dir, f'trained_model_adaptive_{dataset_name}.pkl'),
        os.path.join(results_dir, f'trained_model_adaptive_{dataset_name}_realnvp.pkl'),
        os.path.join(results_dir, f'trained_model_adaptive_{dataset_name}_maf.pkl'),
        os.path.join(results_dir, f'trained_model_adaptive_{dataset_name}_iaf.pkl')
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"❌ No adaptive model files found for {dataset_name}")
        print("Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("Please run the adaptive training script first.")
        return None
    
    print(f"📁 Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data['model']
            flow_losses = saved_data.get('flow_losses', [])
            meta_losses = saved_data.get('meta_losses', [])
            flow_types = saved_data.get('flow_types', ['realnvp', 'maf', 'iaf'])
        else:
            model = saved_data  # Legacy format
            flow_losses = []
            meta_losses = []
            flow_types = ['realnvp', 'maf', 'iaf']
    
    # Ensure model and data are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_data = test_data.to(device)
    
    # CHANGE: Use actual flow types from saved data
    flow_names = flow_types[:len(model.flows)]
    print(f"🔄 Model uses flows: {flow_names}")
    
    # Generate samples for evaluation using adaptive sampling
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(2000)
        
        # Individual flow samples
        flow_samples = {}
        for i, name in enumerate(flow_names):
            flow_samples[name] = model.flows[i].sample(1000)
    
    # Compute overall mixture metrics
    coverage = compute_coverage(test_data, generated_samples)
    quality = compute_quality(test_data, generated_samples)
    diversity_metrics = evaluate_flow_diversity(model)
    
    # For adaptive model, use adaptive mixture log probability
    with torch.no_grad():
        model_output = model.forward(test_data)
        log_prob = model_output['mixture_log_prob'].mean().item()
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names)
    
    # Evaluate meta-learner weights
    meta_learner_metrics = evaluate_meta_learner_weights(model, test_data, flow_names)
    
    # Evaluate adaptive sampling behavior
    adaptive_sampling_metrics = evaluate_adaptive_sampling_behavior(model, n_samples=1000)
    
    # CHANGE: Add direct vs adaptive performance comparison
    direct_vs_adaptive_metrics = evaluate_direct_vs_adaptive_performance(model, test_data)
    
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
        'flow_losses': flow_losses,
        'meta_losses': meta_losses,
        'individual_flow_metrics': individual_flow_metrics,
        'meta_learner_metrics': meta_learner_metrics,
        'adaptive_sampling_metrics': adaptive_sampling_metrics,
        'direct_vs_adaptive_metrics': direct_vs_adaptive_metrics,  # CHANGE: New
        'flow_names': flow_names  # CHANGE: New
    }
    
    print(f"📊 Overall Adaptive Mixture Results for {dataset_name}:")
    print(f"   Coverage: {coverage:.3f}")
    print(f"   Quality: {quality:.3f}")
    print(f"   Log Probability: {log_prob:.3f}")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    
    # CHANGE: Print direct vs adaptive comparison
    if direct_vs_adaptive_metrics:
        print(f"\n🔍 Direct vs Adaptive Performance Comparison:")
        dvm = direct_vs_adaptive_metrics
        if 'adaptive_log_prob' in dvm:
            print(f"   Direct log-prob: {dvm['direct_log_prob']:.4f}")
            print(f"   Adaptive log-prob: {dvm['adaptive_log_prob']:.4f}")
            print(f"   Log-prob improvement: {dvm['log_prob_improvement']:.4f} ({'better' if dvm['log_prob_improvement'] >= 0 else 'worse'})")
            print(f"   Direct coverage: {dvm['direct_coverage']:.3f}")
            print(f"   Adaptive coverage: {dvm['adaptive_coverage']:.3f}")
            print(f"   Coverage improvement: {dvm['coverage_improvement']:.3f} ({'better' if dvm['coverage_improvement'] >= 0 else 'worse'})")
            print(f"   Direct quality: {dvm['direct_quality']:.3f}")
            print(f"   Adaptive quality: {dvm['adaptive_quality']:.3f}")
            print(f"   Quality improvement: {dvm['quality_improvement']:.3f} ({'better' if dvm['quality_improvement'] >= 0 else 'worse'})")
        else:
            print(f"   Direct log-prob: {dvm['direct_log_prob']:.4f}")
            print(f"   Direct coverage: {dvm['direct_coverage']:.3f}")
            print(f"   Direct quality: {dvm['direct_quality']:.3f}")
            print(f"   (Meta-learner not trained - no adaptive comparison)")
    
    # Print meta-learner analysis
    if meta_learner_metrics:
        print(f"\n🧠 Meta-Learner Analysis:")
        print(f"   Weight Entropy (diversity): {meta_learner_metrics['weight_entropy']:.3f}")
        for name, stats in meta_learner_metrics['weight_stats'].items():
            print(f"   {name.upper()} weight - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Print adaptive sampling analysis
    if adaptive_sampling_metrics:
        print(f"\n🎯 Adaptive Sampling Analysis:")
        adaptive_mean = adaptive_sampling_metrics['adaptive_stats']['mean']
        uniform_mean = adaptive_sampling_metrics['uniform_stats']['mean']
        adaptive_std = adaptive_sampling_metrics['adaptive_stats']['std']
        uniform_std = adaptive_sampling_metrics['uniform_stats']['std']
        
        print(f"   Adaptive mean: [{adaptive_mean[0]:.3f}, {adaptive_mean[1]:.3f}]")
        print(f"   Uniform mean:  [{uniform_mean[0]:.3f}, {uniform_mean[1]:.3f}]")
        print(f"   Adaptive std:  [{adaptive_std[0]:.3f}, {adaptive_std[1]:.3f}]")
        print(f"   Uniform std:   [{uniform_std[0]:.3f}, {uniform_std[1]:.3f}]")
    
    # Print individual flow performance
    print(f"\n📊 Individual Flow Results:")
    for name, metrics in individual_flow_metrics.items():
        print(f"   {name.upper()} Flow:")
        print(f"     Coverage: {metrics['coverage']:.3f}")
        print(f"     Quality: {metrics['quality']:.3f}")
        print(f"     Log Probability: {metrics['log_probability']:.3f}")
    
    # Detailed flow analysis
    print(f"\n🔍 Flow Component Analysis:")
    for i, (name, samples) in enumerate(flow_samples.items()):
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        print(f"     {name.upper()} - Mean: [{mean[0]:.2f}, {mean[1]:.2f}], "
              f"Std: [{std[0]:.2f}, {std[1]:.2f}]")
    
    return results

def comprehensive_evaluation():
    """Comprehensive evaluation of all trained Sequential AMF-VI Adaptive models."""
    
    # Define datasets to evaluate
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        results = evaluate_single_dataset(dataset_name)
        if results is not None:
            all_results[dataset_name] = results
    
    if not all_results:
        print("❌ No trained adaptive models found. Please run training first.")
        return None
    
    # Create comprehensive visualization
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print(f"{'='*60}")
    
    n_datasets = len(all_results)
    # CHANGE: Add row for direct vs adaptive comparison
    fig, axes = plt.subplots(4, n_datasets, figsize=(5*n_datasets, 16))
    
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
        
        # Row 2: Generated samples (adaptive)
        plot_samples(results['generated_samples'], f"Adaptive Samples", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Adaptive vs Uniform comparison
        if results['adaptive_sampling_metrics']:
            adaptive_samples = results['adaptive_sampling_metrics']['adaptive_stats']['samples']
            uniform_samples = results['adaptive_sampling_metrics']['uniform_stats']['samples']
            
            # Plot both on same axes with different colors
            plot_samples(adaptive_samples[:500], f"Adaptive vs Uniform", 
                        axes[2, i], color='red', alpha=0.4, s=15)
            uniform_np = uniform_samples[:500].detach().cpu().numpy()
            axes[2, i].scatter(uniform_np[:, 0], uniform_np[:, 1], 
                             alpha=0.4, s=15, c='blue', label='Uniform')
            axes[2, i].legend(['Adaptive', 'Uniform'], fontsize=8)
        else:
            axes[2, i].text(0.5, 0.5, 'No Adaptive\nSampling', ha='center', va='center', 
                          transform=axes[2, i].transAxes)
            axes[2, i].set_title(f"Sampling Comparison")
        
        # CHANGE: Row 4: Direct vs Adaptive comparison
        if results['direct_vs_adaptive_metrics'] and 'adaptive_samples' in results['direct_vs_adaptive_metrics']:
            direct_samples = results['direct_vs_adaptive_metrics']['direct_samples'][:500]
            adaptive_samples = results['direct_vs_adaptive_metrics']['adaptive_samples'][:500]
            
            plot_samples(direct_samples, f"Direct vs Adaptive", 
                        axes[3, i], color='green', alpha=0.4, s=15)
            adaptive_np = adaptive_samples.detach().cpu().numpy()
            axes[3, i].scatter(adaptive_np[:, 0], adaptive_np[:, 1], 
                             alpha=0.4, s=15, c='red', label='Adaptive')
            axes[3, i].legend(['Direct', 'Adaptive'], fontsize=8)
        else:
            axes[3, i].text(0.5, 0.5, 'No Direct\nComparison', ha='center', va='center', 
                          transform=axes[3, i].transAxes)
            axes[3, i].set_title(f"Direct vs Adaptive")
    
    plt.tight_layout()
    plt.suptitle('Sequential AMF-VI Adaptive Sampling Evaluation - All Datasets', fontsize=16, y=0.98)
    
    # Save comprehensive plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'sequential_adaptive_comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # CHANGE: Enhanced summary table with direct vs adaptive comparison
    print(f"\n{'='*90}")
    print("SUMMARY TABLE - DIRECT VS ADAPTIVE COMPARISON")
    print(f"{'='*90}")
    print(f"{'Dataset':<15} | {'Direct Cov':<9} | {'Adapt Cov':<9} | {'Direct Qual':<10} | {'Adapt Qual':<10} | {'Cov Imp':<7} | {'Qual Imp':<7}")
    print("-" * 90)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        dvm = results.get('direct_vs_adaptive_metrics', {})
        
        direct_cov = dvm.get('direct_coverage', 0.0)
        adaptive_cov = dvm.get('adaptive_coverage', results['coverage'])
        direct_qual = dvm.get('direct_quality', 0.0)
        adaptive_qual = dvm.get('adaptive_quality', results['quality'])
        cov_imp = dvm.get('coverage_improvement', 0.0)
        qual_imp = dvm.get('quality_improvement', 0.0)
        
        print(f"{dataset_name:<15} | {direct_cov:<9.3f} | {adaptive_cov:<9.3f} | {direct_qual:<10.3f} | "
              f"{adaptive_qual:<10.3f} | {cov_imp:<7.3f} | {qual_imp:<7.3f}")
        
        summary_data.append([
            dataset_name,
            direct_cov, adaptive_cov, cov_imp,
            direct_qual, adaptive_qual, qual_imp,
            dvm.get('direct_log_prob', 0.0),
            dvm.get('adaptive_log_prob', results['log_probability']),
            dvm.get('log_prob_improvement', 0.0)
        ])
    
    # CHANGE: Save direct vs adaptive comparison to CSV
    with open(os.path.join(results_dir, 'sequential_direct_vs_adaptive_comparison.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'direct_coverage', 'adaptive_coverage', 'coverage_improvement',
                        'direct_quality', 'adaptive_quality', 'quality_improvement',
                        'direct_log_prob', 'adaptive_log_prob', 'log_prob_improvement'])
        writer.writerows(summary_data)
    
    # Original summary table with weight entropy
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - ADAPTIVE MIXTURE RESULTS")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Weight Ent':<10}")
    print("-" * 80)
    
    mixture_summary_data = []
    for dataset_name, results in all_results.items():
        weight_ent = results['meta_learner_metrics']['weight_entropy'] if results['meta_learner_metrics'] else 0.0
        print(f"{dataset_name:<15} | {results['coverage']:<8.3f} | {results['quality']:<8.3f} | "
              f"{results['log_probability']:<10.3f} | {weight_ent:<10.3f}")
        
        mixture_summary_data.append([
            dataset_name,
            results['coverage'],
            results['quality'],
            results['log_probability'],
            results['flow_separation'],
            weight_ent
        ])
    
    # Save mixture metrics to CSV
    with open(os.path.join(results_dir, 'sequential_adaptive_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'coverage', 'quality', 'log_probability', 'flow_separation', 'weight_entropy'])
        writer.writerows(mixture_summary_data)
    
    # Individual flow metrics (unchanged)
    individual_flow_data = []
    
    print(f"\n{'='*80}")
    print("INDIVIDUAL FLOW COMPARISON")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"{'Flow':<8} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Avg Weight':<10}")
        print("-" * 65)
        
        if 'individual_flow_metrics' in results:
            for flow_name, metrics in results['individual_flow_metrics'].items():
                # Get average weight from meta-learner
                avg_weight = 0.0
                if results['meta_learner_metrics']:
                    avg_weight = results['meta_learner_metrics']['weight_stats'][flow_name]['mean']
                
                print(f"{flow_name.upper():<8} | {metrics['coverage']:<8.3f} | {metrics['quality']:<8.3f} | "
                      f"{metrics['log_probability']:<10.3f} | {avg_weight:<10.3f}")
                
                individual_flow_data.append([
                    dataset_name,
                    flow_name,
                    metrics['coverage'],
                    metrics['quality'],
                    metrics['log_probability'],
                    avg_weight
                ])
    
    # Save individual flow metrics to CSV
    with open(os.path.join(results_dir, 'sequential_adaptive_individual_flow_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow_name', 'coverage', 'quality', 'log_probability', 'avg_weight'])
        writer.writerows(individual_flow_data)
    
    # Adaptive sampling comparison (unchanged)
    adaptive_comparison_data = []
    print(f"\n{'='*80}")
    print("ADAPTIVE SAMPLING COMPARISON")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Adaptive Mean':<15} | {'Uniform Mean':<15} | {'Difference':<10}")
    print("-" * 80)
    
    for dataset_name, results in all_results.items():
        if results['adaptive_sampling_metrics']:
            adaptive_mean = results['adaptive_sampling_metrics']['adaptive_stats']['mean']
            uniform_mean = results['adaptive_sampling_metrics']['uniform_stats']['mean']
            mean_diff = np.linalg.norm(adaptive_mean - uniform_mean)
            
            print(f"{dataset_name:<15} | [{adaptive_mean[0]:.2f}, {adaptive_mean[1]:.2f}]     | "
                  f"[{uniform_mean[0]:.2f}, {uniform_mean[1]:.2f}]     | {mean_diff:<10.3f}")
            
            adaptive_comparison_data.append([
                dataset_name,
                adaptive_mean[0], adaptive_mean[1],
                uniform_mean[0], uniform_mean[1],
                mean_diff
            ])
    
    if adaptive_comparison_data:
        with open(os.path.join(results_dir, 'sequential_adaptive_sampling_comparison.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'adaptive_mean_x', 'adaptive_mean_y', 'uniform_mean_x', 'uniform_mean_y', 'mean_difference'])
            writer.writerows(adaptive_comparison_data)
    
    print(f"\n✅ Comprehensive adaptive evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, 'sequential_adaptive_comprehensive_evaluation.png')}")
    print(f"   Mixture metrics saved to: {os.path.join(results_dir, 'sequential_adaptive_comprehensive_metrics.csv')}")
    print(f"   Individual flow metrics saved to: {os.path.join(results_dir, 'sequential_adaptive_individual_flow_metrics.csv')}")
    print(f"   Adaptive sampling comparison saved to: {os.path.join(results_dir, 'sequential_adaptive_sampling_comparison.csv')}")
    print(f"   DIRECT VS ADAPTIVE comparison saved to: {os.path.join(results_dir, 'sequential_direct_vs_adaptive_comparison.csv')}")
    
    # Performance analysis
    best_dataset = max(all_results.items(), key=lambda x: x[1]['coverage'])
    print(f"\n🏆 Best performing dataset (adaptive mixture): {best_dataset[0]} (Coverage: {best_dataset[1]['coverage']:.3f})")
    
    # Best performing individual flow
    best_individual_flow = None
    best_coverage = 0
    for dataset_name, results in all_results.items():
        if 'individual_flow_metrics' in results:
            for flow_name, metrics in results['individual_flow_metrics'].items():
                if metrics['coverage'] > best_coverage:
                    best_coverage = metrics['coverage']
                    best_individual_flow = (dataset_name, flow_name)
    
    if best_individual_flow:
        print(f"🏆 Best performing individual flow: {best_individual_flow[1].upper()} on {best_individual_flow[0]} (Coverage: {best_coverage:.3f})")
    
    # Most balanced meta-learner
    most_balanced = max(all_results.items(), 
                       key=lambda x: x[1]['meta_learner_metrics']['weight_entropy'] if x[1]['meta_learner_metrics'] else 0)
    if most_balanced[1]['meta_learner_metrics']:
        entropy = most_balanced[1]['meta_learner_metrics']['weight_entropy']
        print(f"🎯 Most balanced meta-learner: {most_balanced[0]} (Weight Entropy: {entropy:.3f})")
    
    # Dataset with most effective adaptive sampling
    if adaptive_comparison_data:
        most_adaptive = max(adaptive_comparison_data, key=lambda x: x[5])  # Max mean difference
        print(f"🎯 Most adaptive sampling difference: {most_adaptive[0]} (Mean difference: {most_adaptive[5]:.3f})")
    
    # CHANGE: Analyze direct vs adaptive improvements
    print(f"\n🔍 Direct vs Adaptive Performance Analysis:")
    coverage_improvements = []
    quality_improvements = []
    log_prob_improvements = []
    
    for dataset_name, results in all_results.items():
        dvm = results.get('direct_vs_adaptive_metrics', {})
        if 'coverage_improvement' in dvm:
            coverage_improvements.append((dataset_name, dvm['coverage_improvement']))
            quality_improvements.append((dataset_name, dvm['quality_improvement']))
            log_prob_improvements.append((dataset_name, dvm['log_prob_improvement']))
    
    if coverage_improvements:
        best_cov_improvement = max(coverage_improvements, key=lambda x: x[1])
        best_qual_improvement = max(quality_improvements, key=lambda x: x[1])
        best_logprob_improvement = max(log_prob_improvements, key=lambda x: x[1])
        
        print(f"   Best coverage improvement: {best_cov_improvement[0]} (+{best_cov_improvement[1]:.3f})")
        print(f"   Best quality improvement: {best_qual_improvement[0]} (+{best_qual_improvement[1]:.3f})")
        print(f"   Best log-prob improvement: {best_logprob_improvement[0]} (+{best_logprob_improvement[1]:.4f})")
        
        # Count how many datasets showed improvement
        cov_improved = sum(1 for _, imp in coverage_improvements if imp > 0)
        qual_improved = sum(1 for _, imp in quality_improvements if imp > 0)
        logprob_improved = sum(1 for _, imp in log_prob_improvements if imp > 0)
        
        total_datasets = len(coverage_improvements)
        print(f"   Datasets with coverage improvement: {cov_improved}/{total_datasets}")
        print(f"   Datasets with quality improvement: {qual_improved}/{total_datasets}")
        print(f"   Datasets with log-prob improvement: {logprob_improved}/{total_datasets}")
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_evaluation()