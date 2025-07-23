import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sequential_amf_vi_refactor_flow import SequentialAMFVI  # Updated import
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
    """Evaluate adaptive sampling behavior and effectiveness."""
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

def evaluate_single_dataset(dataset_name):
    """Evaluate a single trained sequential model with quality-aware adaptive sampling."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name.upper()} dataset (Quality-Aware Adaptive)")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    
    # Load pre-trained quality-aware model (UPDATED PATH)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Try different possible model file names
    possible_paths = [
        os.path.join(results_dir, f'trained_model_quality_aware_{dataset_name}.pkl'),
        os.path.join(results_dir, f'trained_model_adaptive_{dataset_name}.pkl'),  # Fallback
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"❌ No model file found for {dataset_name}")
        print("Please run the quality-aware training script first: python sequential_amf_vi_refactor_flow.py")
        return None
    
    print(f"📂 Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data['model']
            flow_losses = saved_data.get('flow_losses', [])
            meta_losses = saved_data.get('meta_losses', [])
            temperature = saved_data.get('temperature', 1.0)
            quality_weight = saved_data.get('quality_weight', 0.1)
        else:
            model = saved_data  # Legacy format
            flow_losses = []
            meta_losses = []
            temperature = 1.0
            quality_weight = 0.1
    
    # Ensure model and data are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_data = test_data.to(device)
    
    # Flow names to match quality-aware version
    flow_names = ['realnvp', 'maf', 'iaf']
    
    # Generate samples for evaluation using quality-aware adaptive sampling
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
    
    # For quality-aware model, use adaptive mixture log probability
    with torch.no_grad():
        model_output = model.forward(test_data)
        log_prob = model_output['mixture_log_prob'].mean().item()
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names[:len(model.flows)])
    
    # Evaluate meta-learner weights
    meta_learner_metrics = evaluate_meta_learner_weights(model, test_data, flow_names[:len(model.flows)])
    
    # Evaluate adaptive sampling behavior
    adaptive_sampling_metrics = evaluate_adaptive_sampling_behavior(model, n_samples=1000)
    
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
        'temperature': temperature,
        'quality_weight': quality_weight
    }
    
    print(f"📊 Overall Quality-Aware Mixture Results for {dataset_name}:")
    print(f"   Coverage: {coverage:.3f}")
    print(f"   Quality: {quality:.3f}")
    print(f"   Log Probability: {log_prob:.3f}")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    print(f"   Temperature: {temperature}")
    print(f"   Quality Weight: {quality_weight}")
    
    # Print meta-learner analysis
    if meta_learner_metrics:
        print(f"\n🧠 Meta-Learner Analysis:")
        print(f"   Weight Entropy (diversity): {meta_learner_metrics['weight_entropy']:.3f}")
        for name, stats in meta_learner_metrics['weight_stats'].items():
            print(f"   {name.upper()} weight - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Print quality scores analysis
    if hasattr(model, 'flow_quality_scores'):
        print(f"\n🏆 Flow Quality Scores:")
        for i, name in enumerate(flow_names[:len(model.flows)]):
            quality_score = model.flow_quality_scores[i].item()
            print(f"   {name.upper()}: {quality_score:.3f}")
    
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
    """Comprehensive evaluation of all trained Sequential AMF-VI Quality-Aware models."""
    
    # Define datasets to evaluate
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        results = evaluate_single_dataset(dataset_name)
        if results is not None:
            all_results[dataset_name] = results
    
    if not all_results:
        print("❌ No trained quality-aware models found. Please run training first.")
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
        
        # Row 2: Generated samples (quality-aware adaptive)
        plot_samples(results['generated_samples'], f"Quality-Aware Samples", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Adaptive vs Uniform comparison
        if results['adaptive_sampling_metrics']:
            adaptive_samples = results['adaptive_sampling_metrics']['adaptive_stats']['samples']
            uniform_samples = results['adaptive_sampling_metrics']['uniform_stats']['samples']
            
            # Plot both on same axes with different colors
            plot_samples(adaptive_samples[:500], f"Quality-Aware vs Uniform", 
                        axes[2, i], color='red', alpha=0.4, s=15)
            uniform_np = uniform_samples[:500].detach().cpu().numpy()
            axes[2, i].scatter(uniform_np[:, 0], uniform_np[:, 1], 
                             alpha=0.4, s=15, c='blue', label='Uniform')
            axes[2, i].legend(['Quality-Aware', 'Uniform'], fontsize=8)
        else:
            axes[2, i].text(0.5, 0.5, 'No Adaptive\nSampling', ha='center', va='center', 
                          transform=axes[2, i].transAxes)
            axes[2, i].set_title(f"Sampling Comparison")
    
    plt.tight_layout()
    plt.suptitle('Sequential AMF-VI Quality-Aware Adaptive Sampling Evaluation - All Datasets', fontsize=16, y=0.98)
    
    # Save comprehensive plot (KEEPING SAME FILENAME FOR COMPARISON)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'sequential_adaptive_comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - QUALITY-AWARE ADAPTIVE MIXTURE RESULTS")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Weight Ent':<10} | {'Temp':<6} | {'Q_Wt':<6}")
    print("-" * 90)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        weight_ent = results['meta_learner_metrics']['weight_entropy'] if results['meta_learner_metrics'] else 0.0
        temp = results.get('temperature', 1.0)
        q_wt = results.get('quality_weight', 0.1)
        
        print(f"{dataset_name:<15} | {results['coverage']:<8.3f} | {results['quality']:<8.3f} | "
              f"{results['log_probability']:<10.3f} | {weight_ent:<10.3f} | {temp:<6.1f} | {q_wt:<6.1f}")
        
        summary_data.append([
            dataset_name,
            results['coverage'],
            results['quality'],
            results['log_probability'],
            results['flow_separation'],
            weight_ent,
            temp,
            q_wt
        ])
    
    # Save mixture metrics to CSV (KEEPING SAME FILENAME FOR COMPARISON)
    with open(os.path.join(results_dir, 'sequential_adaptive_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'coverage', 'quality', 'log_probability', 'flow_separation', 'weight_entropy', 'temperature', 'quality_weight'])
        writer.writerows(summary_data)
    
    # Create individual flow metrics CSV
    individual_flow_data = []
    flow_names = ['realnvp', 'maf', 'iaf']
    
    print(f"\n{'='*80}")
    print("INDIVIDUAL FLOW COMPARISON")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"{'Flow':<8} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Avg Weight':<10} | {'Flow Quality':<12}")
        print("-" * 80)
        
        if 'individual_flow_metrics' in results:
            for i, (flow_name, metrics) in enumerate(results['individual_flow_metrics'].items()):
                # Get average weight from meta-learner
                avg_weight = 0.0
                if results['meta_learner_metrics']:
                    avg_weight = results['meta_learner_metrics']['weight_stats'][flow_name]['mean']
                
                # Get flow quality score
                flow_quality = 0.0
                if hasattr(results['model'], 'flow_quality_scores') and i < len(results['model'].flow_quality_scores):
                    flow_quality = results['model'].flow_quality_scores[i].item()
                
                print(f"{flow_name.upper():<8} | {metrics['coverage']:<8.3f} | {metrics['quality']:<8.3f} | "
                      f"{metrics['log_probability']:<10.3f} | {avg_weight:<10.3f} | {flow_quality:<12.3f}")
                
                individual_flow_data.append([
                    dataset_name,
                    flow_name,
                    metrics['coverage'],
                    metrics['quality'],
                    metrics['log_probability'],
                    avg_weight,
                    flow_quality
                ])
    
    # Save individual flow metrics to CSV (KEEPING SAME FILENAME FOR COMPARISON)
    with open(os.path.join(results_dir, 'sequential_adaptive_individual_flow_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow_name', 'coverage', 'quality', 'log_probability', 'avg_weight', 'flow_quality'])
        writer.writerows(individual_flow_data)
    
    # Create adaptive sampling comparison CSV
    adaptive_comparison_data = []
    print(f"\n{'='*80}")
    print("QUALITY-AWARE ADAPTIVE SAMPLING COMPARISON")
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
        # Save adaptive sampling comparison to CSV (KEEPING SAME FILENAME FOR COMPARISON)
        with open(os.path.join(results_dir, 'sequential_adaptive_sampling_comparison.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'adaptive_mean_x', 'adaptive_mean_y', 'uniform_mean_x', 'uniform_mean_y', 'mean_difference'])
            writer.writerows(adaptive_comparison_data)
    
    print(f"\n✅ Comprehensive quality-aware evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, 'sequential_adaptive_comprehensive_evaluation.png')}")
    print(f"   Mixture metrics saved to: {os.path.join(results_dir, 'sequential_adaptive_comprehensive_metrics.csv')}")
    print(f"   Individual flow metrics saved to: {os.path.join(results_dir, 'sequential_adaptive_individual_flow_metrics.csv')}")
    print(f"   Adaptive sampling comparison saved to: {os.path.join(results_dir, 'sequential_adaptive_sampling_comparison.csv')}")
    
    # Best performing dataset (mixture)
    best_dataset = max(all_results.items(), key=lambda x: x[1]['coverage'])
    print(f"\n🏆 Best performing dataset (quality-aware mixture): {best_dataset[0]} (Coverage: {best_dataset[1]['coverage']:.3f})")
    
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
    
    # Quality-aware specific analysis
    print(f"\n{'='*80}")
    print("QUALITY-AWARE ANALYSIS")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        if hasattr(results['model'], 'flow_quality_scores'):
            quality_scores = [score.item() for score in results['model'].flow_quality_scores]
            temp = results.get('temperature', 1.0)
            q_wt = results.get('quality_weight', 0.1)
            
            print(f"{dataset_name.upper()}:")
            print(f"  Temperature: {temp}, Quality Weight: {q_wt}")
            print(f"  Flow Quality Scores: {[f'{score:.3f}' for score in quality_scores]}")
            print(f"  Highest Quality Flow: {flow_names[np.argmax(quality_scores)].upper()} ({max(quality_scores):.3f})")
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_evaluation()