import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.model import SimpleAMFVI
from amf_vi.loss import SimpleIWAELoss
from amf_vi.utils import create_multimodal_data, multimodal_log_prob, plot_comparison

def train_amf_vi():
    """Train AMF-VI on 2D multimodal data."""
    
    # Create data
    data = create_multimodal_data(2000)
    
    # Create model with different flow types
    model = SimpleAMFVI(
        dim=2, 
        flow_types=['realnvp', 'planar', 'radial'],
        n_components=3
    )
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SimpleIWAELoss(n_importance_samples=5)
    
    # Training loop
    n_epochs = 200
    batch_size = 64
    
    for epoch in range(n_epochs):
        # Mini-batch training
        perm = torch.randperm(len(data))
        total_loss = 0
        
        for i in range(0, len(data), batch_size):
            indices = perm[i:i+batch_size]
            batch = data[indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Compute loss
            main_loss = loss_fn(output, multimodal_log_prob, batch)
            reg_loss = model.regularization_loss()
            loss = main_loss + reg_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    
    # Generate samples and visualize
    model.eval()
    with torch.no_grad():
        # Sample from mixture
        model_samples = model.sample(1000)
        
        # Sample from individual flows
        flow_samples = {}
        flow_names = ['realnvp', 'planar', 'radial']
        for i, name in enumerate(flow_names):
            if i < len(model.flows):
                samples = model.flows[i].sample(1000)
                flow_samples[name] = samples
        
        # Plot comparison
        fig = plot_comparison(data, model_samples, flow_samples)
        plt.show()
        
        # Print some statistics
        print(f"\nResults:")
        print(f"Target data mean: {data.mean(dim=0)}")
        print(f"Model samples mean: {model_samples.mean(dim=0)}")
        print(f"Target data std: {data.std(dim=0)}")
        print(f"Model samples std: {model_samples.std(dim=0)}")

if __name__ == "__main__":
    train_amf_vi()