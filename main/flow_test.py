#!/usr/bin/env python3
"""
Test script to evaluate different flow types and layer configurations.
Tests each flow's ability to fit synthetic 2D data with minimal code.

Usage:
    python main/test_flows.py
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amf_vi.flows import (
    RealNVPFlow, PlanarFlow, RadialFlow, 
    MAFFlow, IAFFlow, GaussianizationFlow
)
from data.data_generator import generate_data

def simple_train(flow, data, epochs=150, lr=1e-3, verbose=False):
    """Simple training loop for a flow."""
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    losses = []
    
    flow.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            # Compute negative log likelihood
            log_prob = flow.log_prob(data)
            loss = -log_prob.mean()
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                if verbose:
                    print(f"  Non-finite loss at epoch {epoch}")
                return losses, False
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            
            optimizer.step()
            losses.append(loss.item())
            
            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:3d}: Loss = {loss.item():.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  Training failed at epoch {epoch}: {e}")
            return losses, False
    
    return losses, True

def evaluate_flow(flow_class, data, layer_configs, flow_name, **flow_kwargs):
    """Evaluate a flow class with different layer configurations."""
    print(f"\n{'='*60}")
    print(f"Testing {flow_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for n_layers in layer_configs:
        print(f"\n--- {n_layers} layers ---")
        
        try:
            # Create flow with appropriate parameters
            if flow_name == "Gaussianization":
                flow = flow_class(dim=2, n_layers=n_layers, **flow_kwargs)
            elif flow_name in ["MAF", "IAF"]:
                flow = flow_class(dim=2, n_layers=n_layers, **flow_kwargs)
            elif flow_name == "RealNVP":
                flow = flow_class(dim=2, n_layers=n_layers, **flow_kwargs)
            else:  # Planar, Radial
                flow = flow_class(dim=2, n_layers=n_layers)
            
            # Count parameters
            n_params = sum(p.numel() for p in flow.parameters())
            print(f"Parameters: {n_params:,}")
            
            # Train
            start_time = time.time()
            losses, success = simple_train(flow, data, epochs=150, verbose=False)
            train_time = time.time() - start_time
            
            if success and losses:
                final_loss = losses[-1]
                print(f"Final loss: {final_loss:.4f}")
                print(f"Train time: {train_time:.1f}s")
                
                # Test sampling
                try:
                    flow.eval()
                    with torch.no_grad():
                        samples = flow.sample(50)
                    sample_success = True
                    print("Sampling: ✓")
                except Exception as e:
                    print(f"Sampling: ✗ ({str(e)[:30]}...)")
                    sample_success = False
                
                # Test log probability computation
                try:
                    with torch.no_grad():
                        test_log_prob = flow.log_prob(data[:50]).mean().item()
                    prob_success = True
                    print("Log prob: ✓")
                except Exception as e:
                    print(f"Log prob: ✗ ({str(e)[:30]}...)")
                    test_log_prob = float('inf')
                    prob_success = False
                
                results[n_layers] = {
                    'final_loss': final_loss,
                    'train_time': train_time,
                    'n_params': n_params,
                    'losses': losses,
                    'sample_success': sample_success,
                    'prob_success': prob_success,
                    'test_log_prob': test_log_prob,
                    'success': True
                }
                
                # Overall status
                if sample_success and prob_success:
                    print("Status: ✓ Full Success")
                elif sample_success or prob_success:
                    print("Status: ⚠ Partial Success")
                else:
                    print("Status: ⚠ Limited Success")
                
            else:
                print("Training: ✗ Failed")
                results[n_layers] = {'success': False, 'final_loss': float('inf')}
                
        except Exception as e:
            print(f"Creation failed: {str(e)[:50]}...")
            results[n_layers] = {'success': False, 'error': str(e)}
    
    return results

def print_summary_table(all_results):
    """Print a summary table of results."""
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    
    # Header
    print(f"{'Flow Type':<15} {'Layers':<7} {'Final Loss':<12} {'Params':<10} {'Time':<8} {'Status':<15}")
    print("-" * 90)
    
    for flow_name, results in all_results.items():
        for n_layers, result in results.items():
            if result.get('success', False):
                final_loss = f"{result['final_loss']:.4f}"
                n_params = f"{result.get('n_params', 0):,}"
                train_time = f"{result.get('train_time', 0):.1f}s"
                
                # Status
                sample_ok = result.get('sample_success', False)
                prob_ok = result.get('prob_success', False)
                if sample_ok and prob_ok:
                    status = "✓ Full"
                elif sample_ok or prob_ok:
                    status = "⚠ Partial"
                else:
                    status = "⚠ Limited"
            else:
                final_loss = "FAILED"
                n_params = "-"
                train_time = "-"
                status = "✗ Failed"
            
            print(f"{flow_name:<15} {n_layers:<7} {final_loss:<12} {n_params:<10} {train_time:<8} {status:<15}")

def find_best_config(all_results):
    """Find the best configuration for each flow type."""
    print(f"\n{'='*60}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*60}")
    
    for flow_name, results in all_results.items():
        successful_results = [(n_layers, result) for n_layers, result in results.items() 
                             if result.get('success', False)]
        
        if successful_results:
            # Find best by final loss
            best_layers, best_result = min(successful_results, key=lambda x: x[1]['final_loss'])
            
            print(f"\n{flow_name}:")
            print(f"  Best layers: {best_layers}")
            print(f"  Final loss: {best_result['final_loss']:.4f}")
            print(f"  Parameters: {best_result.get('n_params', 0):,}")
            print(f"  Train time: {best_result.get('train_time', 0):.1f}s")
            
            # Performance analysis
            loss = best_result['final_loss']
            if loss < 0.5:
                perf = "Excellent"
            elif loss < 1.0:
                perf = "Very Good"
            elif loss < 2.0:
                perf = "Good"
            elif loss < 3.0:
                perf = "Fair"
            else:
                perf = "Poor"
            print(f"  Performance: {perf}")
            
            # Efficiency analysis
            params = best_result.get('n_params', 0)
            if params < 1000:
                eff = "Very Efficient"
            elif params < 5000:
                eff = "Efficient"
            elif params < 20000:
                eff = "Moderate"
            else:
                eff = "Heavy"
            print(f"  Efficiency: {eff} ({params:,} params)")
        else:
            print(f"\n{flow_name}: No successful configurations")

def plot_comparison(all_results, data):
    """Create comparison plots."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Color scheme for flows
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    flow_colors = {}
    
    # 1. Training curves
    ax = axes[0, 0]
    color_idx = 0
    for flow_name, results in all_results.items():
        if flow_name not in flow_colors:
            flow_colors[flow_name] = colors[color_idx % len(colors)]
            color_idx += 1
        
        for n_layers, result in results.items():
            if result.get('success', False) and 'losses' in result:
                losses = result['losses']
                ax.plot(losses, label=f"{flow_name} ({n_layers}L)", 
                       color=flow_colors[flow_name], alpha=0.7, linewidth=1.5)
    
    ax.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log Likelihood')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Final loss vs layers
    ax = axes[0, 1]
    for flow_name, results in all_results.items():
        layers = []
        final_losses = []
        for n_layers, result in results.items():
            if result.get('success', False):
                layers.append(n_layers)
                final_losses.append(result['final_loss'])
        if layers:
            ax.plot(layers, final_losses, 'o-', label=flow_name, 
                   color=flow_colors[flow_name], markersize=6, linewidth=2)
    
    ax.set_title('Final Loss vs Layers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Final Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Parameters vs layers  
    ax = axes[0, 2]
    for flow_name, results in all_results.items():
        layers = []
        n_params = []
        for n_layers, result in results.items():
            if result.get('success', False) and 'n_params' in result:
                layers.append(n_layers)
                n_params.append(result['n_params'])
        if layers:
            ax.plot(layers, n_params, 's-', label=flow_name, 
                   color=flow_colors[flow_name], markersize=6, linewidth=2)
    
    ax.set_title('Parameters vs Layers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Number of Parameters')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Training time vs layers
    ax = axes[1, 0]
    for flow_name, results in all_results.items():
        layers = []
        times = []
        for n_layers, result in results.items():
            if result.get('success', False) and 'train_time' in result:
                layers.append(n_layers)
                times.append(result['train_time'])
        if layers:
            ax.plot(layers, times, '^-', label=flow_name, 
                   color=flow_colors[flow_name], markersize=6, linewidth=2)
    
    ax.set_title('Training Time vs Layers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Training Time (seconds)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Data scatter
    ax = axes[1, 1]
    data_np = data.numpy()
    ax.scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=20, c='darkblue')
    ax.set_title('Target Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(True, alpha=0.3)
    
    # 6. Success rate and efficiency
    ax = axes[1, 2]
    flow_names = list(all_results.keys())
    success_rates = []
    
    for flow_name in flow_names:
        results = all_results[flow_name]
        total = len(results)
        successful = sum(1 for r in results.values() if r.get('success', False))
        success_rates.append(successful / total * 100 if total > 0 else 0)
    
    bars = ax.bar(flow_names, success_rates, alpha=0.7, 
                  color=[flow_colors.get(name, 'gray') for name in flow_names])
    ax.set_title('Success Rate by Flow Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Flow Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    """Main test function."""
    print("Flow Performance Testing")
    print("=" * 60)
    
    # Test different datasets
    available_datasets = ['two_moons', 'multimodal', 'rings', 'banana', 'x_shape']
    dataset_name = 'two_moons'  # Change this to test different datasets
    
    print(f"Available datasets: {available_datasets}")
    data = generate_data(dataset_name, n_samples=800, noise=0.1)
    print(f"Using dataset: {dataset_name}")
    print(f"Data shape: {data.shape}")
    print(f"Data range: X1=[{data[:, 0].min():.2f}, {data[:, 0].max():.2f}], "
          f"X2=[{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
    
    # Test configurations
    layer_configs = [2, 4, 6, 8]  # Number of layers to test
    
    # Flow configurations with optimized parameters
    flow_configs = [
        (RealNVPFlow, "RealNVP", {'hidden_dim': 32}),
        (PlanarFlow, "Planar", {}),
        (RadialFlow, "Radial", {}),
        (MAFFlow, "MAF", {'hidden_dim': 32}),
        (IAFFlow, "IAF", {'hidden_dim': 32}),
        (GaussianizationFlow, "Gaussianization", {'n_anchors': 25, 'n_reflections': 3}),
    ]
    
    # Run tests
    all_results = {}
    
    for flow_class, flow_name, kwargs in flow_configs:
        try:
            results = evaluate_flow(flow_class, data, layer_configs, flow_name, **kwargs)
            all_results[flow_name] = results
        except Exception as e:
            print(f"Failed to test {flow_name}: {e}")
            all_results[flow_name] = {}
    
    # Generate summary
    print_summary_table(all_results)
    find_best_config(all_results)
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    successful_flows = []
    for flow_name, results in all_results.items():
        best_loss = min([r.get('final_loss', float('inf')) for r in results.values() 
                        if r.get('success', False)], default=float('inf'))
        if best_loss < float('inf'):
            successful_flows.append((flow_name, best_loss))
    
    if successful_flows:
        successful_flows.sort(key=lambda x: x[1])
        print(f"Best performing flow: {successful_flows[0][0]} (loss: {successful_flows[0][1]:.4f})")
        
        if len(successful_flows) > 1:
            print("Top 3 flows by performance:")
            for i, (name, loss) in enumerate(successful_flows[:3], 1):
                print(f"  {i}. {name}: {loss:.4f}")
    
    # Create and save plots
    try:
        fig = plot_comparison(all_results, data)
        
        # Save plots
        os.makedirs('results', exist_ok=True)
        fig.savefig('results/flow_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: results/flow_comparison.png")
        
        # Show plots if in interactive environment
        try:
            plt.show()
        except:
            print("Note: Run in interactive environment to display plots")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    total_flows = len(flow_configs)
    successful_flows_count = sum(1 for results in all_results.values() 
                                if any(r.get('success', False) for r in results.values()))
    print(f"Flows tested: {total_flows}")
    print(f"Flows with successful configs: {successful_flows_count}")
    print(f"Dataset used: {dataset_name}")
    print("Check 'results/flow_comparison.png' for detailed plots")
    
    return all_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = main()