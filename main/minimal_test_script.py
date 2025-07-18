#!/usr/bin/env python3
"""
Minimal test script to verify the inplace operation fix.
Run this after updating the flow files to test if the error is resolved.
"""

import torch
import torch.optim as optim
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_generator import generate_data

# Test with just one flow type at a time
def test_single_flow(flow_class, flow_name):
    """Test a single flow to isolate the inplace operation issue."""
    
    print(f"🧪 Testing {flow_name} flow...")
    
    # Generate simple test data
    data = generate_data('banana', n_samples=100)
    device = torch.device('cpu')  # Use CPU to avoid GPU memory issues
    data = data.to(device)
    
    # Create flow
    flow = flow_class(dim=2, n_layers=2)  # Smaller for faster testing
    flow = flow.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    
    # Enable anomaly detection to catch inplace operations
    torch.autograd.set_detect_anomaly(True)
    
    # Test training loop
    success = True
    try:
        for epoch in range(5):  # Just 5 epochs for quick test
            optimizer.zero_grad()
            
            # Forward pass
            log_prob = flow.log_prob(data)
            loss = -log_prob.mean()
            
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Backward pass - this is where inplace errors occur
            loss.backward()
            optimizer.step()
            
        print(f"✅ {flow_name} flow test PASSED")
        
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"❌ {flow_name} flow test FAILED - Inplace operation error:")
            print(f"   {e}")
            success = False
        else:
            print(f"❌ {flow_name} flow test FAILED - Other error:")
            print(f"   {e}")
            success = False
    
    finally:
        torch.autograd.set_detect_anomaly(False)
    
    return success

def test_realnvp():
    """Test RealNVP (should work as baseline)."""
    try:
        from amf_vi.flows.realnvp import RealNVPFlow
        return test_single_flow(RealNVPFlow, "RealNVP")
    except ImportError as e:
        print(f"❌ Could not import RealNVP: {e}")
        return False

def test_maf():
    """Test MAF (the main problematic flow)."""
    try:
        from amf_vi.flows.maf import MAFFlow
        return test_single_flow(MAFFlow, "MAF")
    except ImportError as e:
        print(f"❌ Could not import MAF: {e}")
        return False

def test_iaf():
    """Test IAF (secondary problematic flow)."""
    try:
        from amf_vi.flows.iaf import IAFFlow
        return test_single_flow(IAFFlow, "IAF")
    except ImportError as e:
        print(f"❌ Could not import IAF: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting minimal inplace operation fix verification")
    print("=" * 60)
    
    results = {}
    
    # Test each flow individually
    print("\n📋 Testing individual flows:")
    results['realnvp'] = test_realnvp()
    results['maf'] = test_maf()
    results['iaf'] = test_iaf()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    for flow_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {flow_name.upper():8} | {status}")
    
    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The inplace operation fix is working.")
        print("   You can now run the full sequential training script.")
    else:
        failed_flows = [name for name, success in results.items() if not success]
        print(f"\n⚠️  Some flows still have issues: {failed_flows}")
        print("   Check the error messages above and apply additional fixes.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)