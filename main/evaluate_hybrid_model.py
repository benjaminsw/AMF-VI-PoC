import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sequential_amf_vi_hybrid import SequentialAMFVI
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
    if not hasattr(model, 'meta_trained') or not model.meta_trained:
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

def evaluate_training_progression(saved_data):
    """Evaluate the three-stage training progression (NEW)."""
    flow_losses = saved_data.get('flow_losses', [])
    meta_losses = saved_data.get('meta_losses', [])
    joint_losses = saved_data.get('joint_losses', [])  # NEW
    
    progression_metrics = {
        'stage1_epochs': len(flow_losses[0]) if flow_losses else 0,
        'stage2_epochs': len(meta_losses),
        'stage3_epochs': len(joint_losses),  # NEW
        'total_epochs': 0,
        'stage1_improvement': 0,
        'stage2_improvement': 0,
        'stage3_improvement': 0  # NEW
    }
    
    # Calculate training improvements
    if flow_losses:
        progression_metrics['total_epochs'] += len(flow_losses[0])
        if len(flow_losses[0]) > 10:
            initial_loss = np.mean(flow_losses[0][:10])
            final_loss = np.mean(flow_losses[0][-10:])
            progression_metrics['stage1_improvement'] = initial_loss - final_loss
    
    if meta_losses:
        progression_metrics['total_epochs'] += len(meta_losses)
        if len(meta_losses) > 10:
            initial_loss = np.mean(meta_losses[:10])
            final_loss = np.mean(meta_losses[-10:])
            progression_metrics['stage2_improvement'] = initial_loss - final_loss
    
    # NEW: Joint training analysis
    if joint_losses:
        progression_metrics['total_epochs'] += len(joint_losses)
        if len(joint_losses) > 10:
            initial_loss = np.mean(joint_losses[:10])
            final_loss = np.mean(joint_losses[-10:])
            progression_metrics['stage3_improvement'] = initial_loss - final_loss
    
    return progression_metrics

def evaluate_single_dataset(dataset_name):
    """Evaluate a single trained sequential hybrid model."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name.upper()} dataset (Hybrid Training)")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    
    # Load pre-trained model (UPDATED PATH)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    model_path = os.path.join(results_dir, f'trained_model_hybrid_{dataset_name}.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        print("Please run the training script first: python sequential_amf_vi_hybrid.py")
        return None
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data['model']
            flow_losses = saved_data.get('flow_losses', [])
            meta_losses = saved_data.get('meta_losses', [])
            joint_losses = saved_data.get('joint_losses', [])  # NEW
        else:
            model = saved_data  # Legacy format
            flow_losses = []
            meta_losses = []
            joint_losses = []
    
    # Ensure model and data are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_data = test_data.to(device)
    
    # Updated flow names to match hybrid version
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
    
    # For hybrid model, use adaptive mixture log probability (UPDATED)
    with torch.no_grad():
        model_output = model.forward(test_data)
        log_prob = model_output['mixture_log_prob'].mean().item()
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names[:len(model.flows)])
    
    # Evaluate meta-learner weights
    meta_learner_metrics = evaluate_meta_learner_weights(model, test_data, flow_names[:len(model.flows)])
    
    # Evaluate training progression (NEW)
    progression_metrics = evaluate_training_progression(saved_data)
    
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
        'joint_losses': joint_losses,  # NEW
        'individual_flow_metrics': individual_flow_metrics,
        'meta_learner_metrics': meta_learner_metrics,
        'progression_metrics': progression_metrics  # NEW
    }
    
    print(f"📊 Overall Mixture Results for {dataset_name}:")
    print(f"   Coverage: {coverage:.3f}")
    print(f"   Quality: {quality:.3f}")
    print(f"   Log Probability: {log_prob:.3f}")
    print(f"   Flow Separation: {diversity_metrics['avg_separation']:.3f}")
    
    # Print meta-learner analysis
    if meta_learner_metrics:
        print(f"\n🧠 Meta-Learner Analysis:")
        print(f"   Weight Entropy (diversity): {meta_learner_metrics['weight_entropy']:.3f}")
        for name, stats in meta_learner_metrics['weight_stats'].items():
            print(f"   {name.upper()} weight - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Print training progression analysis (NEW)
    print(f"\n🏁 Training Progression Analysis:")
    print(f"   Stage 1 (Flows): {progression_metrics['stage1_epochs']} epochs, "
          f"Improvement: {progression_metrics['stage1_improvement']:.3f}")
    print(f"   Stage 2 (Meta-learner): {progression_metrics['stage2_epochs']} epochs, "
          f"Improvement: {progression_metrics['stage2_improvement']:.3f}")
    print(f"   Stage 3 (Joint): {progression_metrics['stage3_epochs']} epochs, "
          f"Improvement: {progression_metrics['stage3_improvement']:.3f}")
    print(f"   Total training: {progression_metrics['total_epochs']} epochs")
    
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
    """Comprehensive evaluation of all trained Sequential AMF-VI Hybrid models."""
    
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
        plot_samples(results['generated_samples'], f"Generated Samples", 
                    axes[1, i], color='red', alpha=0.6)
        
        # Row 3: Training progression bar chart (NEW)
        stages = ['Stage 1', 'Stage 2', 'Stage 3']
        epochs = [
            results['progression_metrics']['stage1_epochs'],
            results['progression_metrics']['stage2_epochs'],
            results['progression_metrics']['stage3_epochs']
        ]
        improvements = [
            results['progression_metrics']['stage1_improvement'],
            results['progression_metrics']['stage2_improvement'],
            results['progression_metrics']['stage3_improvement']
        ]
        
        ax3 = axes[2, i]
        x_pos = np.arange(len(stages))
        bars = ax3.bar(x_pos, epochs, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax3.set_title(f"Training Stages")
        ax3.set_xlabel('Training Stage')
        ax3.set_ylabel('Epochs')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(stages)
        
        # Add improvement text on bars
        for j, (bar, improvement) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{improvement:.2f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Sequential AMF-VI Hybrid Evaluation - All Datasets', fontsize=16, y=0.98)
    
    # Save comprehensive plot (UPDATED FILENAME)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'sequential_hybrid_comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - HYBRID MIXTURE RESULTS")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'Coverage':<8} | {'Quality':<8} | {'Log Prob':<10} | {'Total Epochs':<12}")
    print("-" * 80)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        total_epochs = results['progression_metrics']['total_epochs']
        print(f"{dataset_name:<15} | {results['coverage']:<8.3f} | {results['quality']:<8.3f} | "
              f"{results['log_probability']:<10.3f} | {total_epochs:<12}")
        
        summary_data.append([
            dataset_name,
            results['coverage'],
            results['quality'],
            results['log_probability'],
            results['flow_separation'],
            results['meta_learner_metrics']['weight_entropy'] if results['meta_learner_metrics'] else 0.0,
            total_epochs
        ])
    
    # Save mixture metrics to CSV (UPDATED FILENAME)
    with open(os.path.join(results_dir, 'sequential_hybrid_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'coverage', 'quality', 'log_probability', 'flow_separation', 'weight_entropy', 'total_epochs'])
        writer.writerows(summary_data)
    
    # Create individual flow metrics CSV
    individual_flow_data = []
    flow_names = ['realnvp', 'maf', 'iaf']
    
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
    
    # Save individual flow metrics to CSV (UPDATED FILENAME)
    with open(os.path.join(results_dir, 'sequential_hybrid_individual_flow_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'flow_name', 'coverage', 'quality', 'log_probability', 'avg_weight'])
        writer.writerows(individual_flow_data)
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"   Results saved to: {os.path.join(results_dir, 'sequential_hybrid_comprehensive_evaluation.png')}")
    print(f"   Mixture metrics saved to: {os.path.join(results_dir, 'sequential_hybrid_comprehensive_metrics.csv')}")
    print(f"   Individual flow metrics saved to: {os.path.join(results_dir, 'sequential_hybrid_individual_flow_metrics.csv')}")
    
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
    
    # Most balanced meta-learner
    most_balanced = max(all_results.items(), 
                       key=lambda x: x[1]['meta_learner_metrics']['weight_entropy'] if x[1]['meta_learner_metrics'] else 0)
    if most_balanced[1]['meta_learner_metrics']:
        entropy = most_balanced[1]['meta_learner_metrics']['weight_entropy']
        print(f"🎯 Most balanced meta-learner: {most_balanced[0]} (Weight Entropy: {entropy:.3f})")
    
    # Most efficient training (NEW)
    most_efficient = min(all_results.items(), key=lambda x: x[1]['progression_metrics']['total_epochs'])
    total_epochs = most_efficient[1]['progression_metrics']['total_epochs']
    print(f"⚡ Most efficient training: {most_efficient[0]} ({total_epochs} total epochs)")
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_evaluation()