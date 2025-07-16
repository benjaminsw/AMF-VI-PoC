import torch
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.planar import PlanarFlow
from amf_vi.flows.radial import RadialFlow
from amf_vi.utils import create_multimodal_data, create_ring_data, plot_samples

def compare_flows_on_different_data():
    """Compare different flow types on various datasets."""
    
    # Create different datasets
    datasets = {
        'multimodal': create_multimodal_data(1000),
        'rings': create_ring_data(1000)
    }
    
    # Create different flows
    flows = {
        'realnvp': RealNVPFlow(2, n_layers=4),
        'planar': PlanarFlow(2, n_layers=8),
        'radial': RadialFlow(2, n_layers=8)
    }
    
    for data_name, data in datasets.items():
        print(f"\nTesting on {data_name} dataset...")
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot original data
        plot_samples(data, f"Original {data_name}", axes[0])
        
        # Test each flow
        for i, (flow_name, flow) in enumerate(flows.items()):
            print(f"  Testing {flow_name} flow...")
            
            # Simple training loop
            optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
            
            for epoch in range(100):
                optimizer.zero_grad()
                
                # Compute negative log likelihood
                log_prob = flow.log_prob(data)
                loss = -log_prob.mean()
                
                loss.backward()
                optimizer.step()
                
                if epoch % 25 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Generate samples
            flow.eval()
            with torch.no_grad():
                samples = flow.sample(1000)
                plot_samples(samples, f"{flow_name.title()}", axes[i+1])
        
        plt.suptitle(f"Flow Comparison on {data_name.title()} Data")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    compare_flows_on_different_data()