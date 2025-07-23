import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sequential_amf_vi import SequentialAMFVI
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

def evaluate_single_dataset(dataset_name, model_type="quality", initial_temp=2.0, final_temp=0.8):
    """Evaluate a single trained sequential model with flexible model type detection."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name.upper()} dataset ({model_type} model)")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    
    # Load pre-trained model with flexible suffix detection
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Try different model file patterns
    possible_suffixes = [
        f"temp_{initial_temp}_{final_temp}",  # Temperature-controlled
        "quality",                            # Quality-based
        "uniform"                            # Uniform weights
    ]
    
    model_path = None
    for suffix in possible_suffixes:
        candidate_path = os.path.join(results_dir, f'trained_model_{dataset_name}_{suffix}.pkl')
        if os.path.exists(candidate_path):
            model_path = candidate_path
            break
    
    if model_path is None:
        print(f"❌ No model file found for {dataset_name}")
        print("Please run the training script first")
        return None
    
    print(f"📂 Loading model from: {os.path.basename(model_path)}")
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data['model']
            losses = saved_data.get('losses', [])
            use_quality_weights_saved = saved_data.get('use_quality_weights', True)
            initial_temperature = saved_data.get('initial_temperature', 2.0)
            final_temperature = saved_data.get('final_temperature', 0.8)
        else:
            model = saved_data  # Legacy format
            losses = []
            use_quality_weights_saved = True
            initial_temperature = 2.0
            final_temperature = 0.8
    
    # Ensure model and data are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_data = test_data.to(device)
    
    # Updated flow names to match amf_vi_quality_weights.py
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
    
    # For sequential model, approximate log probability using flow predictions
    with torch.no_grad():
        flow_predictions = model.get_flow_predictions(test_data)
        log_prob = torch.logsumexp(flow_predictions, dim=1).mean().item()
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names[:len(model.flows)])
    
    # Get flow weights and temperature info if using quality weights
    flow_weights = None
    current_temperature = None
    if hasattr(model, 'use_quality_weights') and model.use_quality_weights and hasattr(model, 'compute_flow_quality_weights'):
        flow_weights = model.compute_flow_quality_weights()
        if hasattr(model, 'current_temperature'):
            current_temperature = model.current_temperature
    
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
        'use_quality_weights': use_quality_weights_saved,
        'flow_weights': flow_weights,
        'current_temperature': current_temperature,
        'initial_temperature': initial_temperature,
        'final_temperature': final_temperature
    }
    
    # Determine weight type for display
    if use_quality_weights_saved:
        if current_temperature is not None:
            weight_type = f"T-Controlled ({initial_temperature:.1f}→{final_temperature:.1f})"
        else:
            weight_type = "Quality-Based"
    else:
        weight_type = "Uniform"
    
    print(f"📊 Overall Mixture Results for {dataset_name} ({weight_type}):")
    print(f"   Coverage: {coverage:.3f}")
    print(f"   Quality: {quality:.3f}")
    print(f"   Log Probability: {log_prob:.3f}")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    
    if flow_weights is not None:
        temp_info = f" (T={current_temperature:.3f})" if current_temperature is not None else ""
        print(f"   Flow Quality Weights{temp_info}: {flow_weights.cpu().numpy()}")
    
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
        weight_info = ""
        if flow_weights is not None and i < len(flow_weights):
            weight_info = f", Weight: {flow_weights[i]:.3f}"
        print(f"     {name.upper()} - Mean: [{mean[0]:.2f}, {mean[1]:.2f}], "
              f"Std: [{std[0]:.2f}, {std[1]:.2f}]{weight_info}")
    
    return results

def comprehensive_evaluation(model_type="temperature", initial_temp=2.0, final_temp=0.8):
    """Comprehensive evaluation of all trained Sequential AMF-VI models."""
    
    # Define datasets to evaluate
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        results = evaluate_single_dataset(dataset_name, model_type, initial_temp, final_temp)
        if results is not None:
            all_results[dataset_name] = results
    
    if not all_results:
        print("❌ No trained models found. Please run training first.")
        return None
    
    # Determine display title based on first model found
    first_result = next(iter(all_results.values()))
    if first_result['use_quality_weights']:
        if first_result['current_temperature'] is not None:
            weight_type = f"T-Controlled ({first_result['initial_temperature']:.1f}→{first_result['final_temperature']:.1f})"
        else:
            weight_type = "Quality-Based"
    else:
        weight_type = "Uniform"
    
    # Create comprehensive visualization
    print(f"\n{'='*60}")
    print(f"CREATING COMPREHENSIVE VISUALIZATION ({weight_type} Weights)")
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
        title_suffix = f" ({weight_type})"
        plot_samples(results['generated_samples'], f"Generated Samples{title_suffix}", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Flow separation matrix
        pairwise_dists = results['diversity_metrics']['pairwise_distances'].detach().cpu().numpy()
        im = axes[2, i].imshow(pairwise_dists, cmap='viridis')
        axes[2, i].set_title(f"Flow Separation")
        plt.colorbar(im, ax=axes[2, i])
    
    plt.tight_layout()
    plt.suptitle(f'Sequential AMF-VI Evaluation Results - All Datasets ({weight_type})', fontsize=16, y=0.98)
    
    # Save comprehensive plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename based on model type
    if weight_type.startswith("T-Controlled"):
        filename_suffix = f"temp_{initial_temp}_{final_temp}"
    elif weight_type == "Quality-Based":
        filename_suffix = "quality"
    else:
        filename_suffix = "uniform"
    
    plot_filename = f'sequential_comprehensive_evaluation_{filename_suffix}.png'
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE - MIXTURE RESULTS ({weight_type} Weights)")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Flow Sep':<8}")
    if any(r['flow_weights'] is not None for r in all_results.values()):
        print(f"{'':<15} | {'':<8} | {'':<8} | {'':<10} | {'':<8} | {'Flow Weights'}")
    print("-" * 80)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        base_info = f"{dataset_name:<15} | {results['coverage']:<8.3f} | {results['quality']:<8.3f} | " \
                   f"{results['log_probability']:<10.3f} | {results['flow_separation']:<8.3f}"
        
        if results.get('flow_weights') is not None:
            weights_str = str(results['flow_weights'].cpu().numpy())
            temp_info = f" (T={results['current_temperature']:.3f})" if results.get('current_temperature') else ""
            print(f"{base_info} | {weights_str}{temp_info}")
        else:
            print(base_info)
        
        summary_data.append([
            dataset_name,
            results['coverage'],
            results['quality'],
            results['log_probability'],
            results['flow_separation'],
            results['use_quality_weights'],
            results.get('current_temperature', ''),
            results.get('initial_temperature', ''),
            results.get('final_temperature', '')
        ])
    
    # Save mixture metrics to CSV
    csv_filename = f'sequential_comprehensive_metrics_{filename_suffix}.csv'
    with open(os.path.join(results_dir, csv_filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'coverage', 'quality', 'log_probability', 'flow_separation', 
                        'use_quality_weights', 'current_temperature', 'initial_temperature', 'final_temperature'])
        writer.writerows(summary_data)
    
    # Create individual flow metrics CSV
    individual_flow_data = []
    flow_names = ['realnvp', 'maf', 'iaf']
    
    print(f"\n{'='*80}")
    print(f"INDIVIDUAL FLOW COMPARISON ({weight_type} Weights)")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"{'Flow':<8} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10}")
        if results.get('flow_weights') is not None:
            print(f"{'':<8} | {'':<8} | {'':<8} | {'':<10} | {'Weight'}")
        print("-" * 50)
        
        if 'individual_flow_metrics' in results:
            for j, (flow_name, metrics) in enumerate(results['individual_flow_metrics'].items()):
                base_info = f"{flow_name.upper():<8} | {metrics['coverage']:<8.3f} | {metrics['quality']:<8.3f} | " \
                           f"{metrics['log_probability']:<10.3f}"
                
                if results.get('flow_weights') is not None and j < len(results['flow_weights']):
                    weight_val = results['flow_weights'][j].item()
                    print(f"{base_info} | {weight_val:<8.3f}")
                else:
                    print(base_info)
                
                individual_flow_data.append([
                    dataset_name,
                    flow_name,
                    metrics['coverage'],
                    metrics['quality'],
                    metrics['log_probability'],
                    results['use_quality_weights'],
                    results.get('current_temperature', ''),
                    weight_val if results.get('flow_weights') is not None and j < len(results['flow_weights']) else ''
                ])
    
    # Save individual flow metrics to CSV
    individual_csv_filename = f'sequential_individual_flow_metrics_{filename_suffix}.csv'
    with open(os.path.join(results_dir, individual_csv_filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow_name', 'coverage', 'quality', 'log_probability', 
                        'use_quality_weights', 'current_temperature', 'flow_weight'])
        writer.writerows(individual_flow_data)
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, plot_filename)}")
    print(f"   Mixture metrics saved to: {os.path.join(results_dir, csv_filename)}")
    print(f"   Individual flow metrics saved to: {os.path.join(results_dir, individual_csv_filename)}")
    
    # Best performing dataset (mixture)
    best_dataset = max(all_results.items(), key=lambda x: x[1]['coverage'])
    print(f"\n🏆 Best performing dataset (mixture): {best_dataset[0]} (Coverage: {best_dataset[1]['coverage']:.3f})")
    
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
    
    return all_results

if __name__ == "__main__":
    # Evaluate temperature-controlled models by default
    # Options: "temperature", "quality", "uniform"
    results = comprehensive_evaluation(model_type="temperature", initial_temp=2.0, final_temp=0.8)