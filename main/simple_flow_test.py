#!/usr/bin/env python3
"""
Minimal flow testing script - quick comparison of flow performance.
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amf_vi.flows import (
    RealNVPFlow, PlanarFlow, RadialFlow, 
    MAFFlow, IAFFlow, GaussianizationFlow
)
from data.data_generator import generate_data

def quick_train(flow, data, epochs=100):
    """Quick training with minimal error handling."""
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        try:
            loss = -flow.log_prob(data).mean()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                optimizer.step()
            else:
                return float('inf'), False
        except:
            return float('inf'), False
    
    return loss.item(), True

def test_flow_config(flow_class, flow_name, data, layers=4, **kwargs):
    """Test a single flow configuration."""
    try:
        # Create and train flow
        flow = flow_class(dim=2, n_layers=layers, **kwargs)
        n_params = sum(p.numel() for p in flow.parameters())
        
        final_loss, success = quick_train(flow, data)
        
        # Test basic functionality
        can_sample = False
        can_logprob = False
        
        if success:
            try:
                flow.eval()
                with torch.no_grad():
                    _ = flow.sample(10)
                can_sample = True
            except:
                pass
            
            try:
                with torch.no_grad():
                    _ = flow.log_prob(data[:10])
                can_logprob = True
            except:
                pass
        
        return {
            'success': success,
            'loss': final_loss,
            'params': n_params,
            'can_sample': can_sample,
            'can_logprob': can_logprob
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)[:50],
            'loss': float('inf'),
            'params': 0,
            'can_sample': False,
            'can_logprob': False
        }

def main():
    """Quick flow comparison."""
    print("=== Quick Flow Performance Test ===\n")
    
    # Generate test data
    data = generate_data('two_moons', n_samples=500)
    print(f"Data: {data.shape[0]} samples from 'two_moons' dataset\n")
    
    # Flow configurations
    flows = [
        (RealNVPFlow, "RealNVP", {'hidden_dim': 32}),
        (PlanarFlow, "Planar", {}),
        (RadialFlow, "Radial", {}),
        (MAFFlow, "MAF", {'hidden_dim': 32}),
        (IAFFlow, "IAF", {'hidden_dim': 32}),
        (GaussianizationFlow, "Gaussianization", {'n_anchors': 20}),
    ]
    
    results = []
    
    print(f"{'Flow':<15} {'Loss':<8} {'Params':<8} {'Sample':<7} {'LogProb':<8} {'Status'}")
    print("-" * 65)
    
    for flow_class, name, kwargs in flows:
        result = test_flow_config(flow_class, name, data, layers=4, **kwargs)
        results.append((name, result))
        
        if result['success']:
            loss_str = f"{result['loss']:.3f}"
            params_str = f"{result['params']:,}"
            sample_str = "✓" if result['can_sample'] else "✗"
            logprob_str = "✓" if result['can_logprob'] else "✗"
            status = "Success"
        else:
            loss_str = "FAIL"
            params_str = "-"
            sample_str = "✗"
            logprob_str = "✗"
            status = "Failed"
        
        print(f"{name:<15} {loss_str:<8} {params_str:<8} {sample_str:<7} {logprob_str:<8} {status}")
    
    # Find best performing flow
    successful = [(name, r) for name, r in results if r['success']]
    if successful:
        best_name, best_result = min(successful, key=lambda x: x[1]['loss'])
        print(f"\nBest performing: {best_name} (loss: {best_result['loss']:.3f})")
    else:
        print("\nNo flows succeeded!")
    
    print("\n=== Test Complete ===")
    return results

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()