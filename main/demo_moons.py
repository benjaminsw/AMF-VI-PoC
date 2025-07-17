import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.model import SimpleAMFVI
from amf_vi.loss import SimpleIWAELoss
from amf_vi.utils import create_two_moons_data, plot_samples

def train_on_moons():
    """Train AMF-VI on two moons dataset."""
    
    # Create two moons data
    data = create_two_moons_data(1500, noise=0.1)
    
    # Create model
    model = SimpleAMFVI(
        dim=2,
        flow_types=['realnvp', 'planar'],  # Only 2 flows for 2 moons
        n_components=2
    )
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SimpleIWAELoss(n_importance_samples=5)
    
    # Training loop
    losses = []
    for epoch in range(300):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss (no target log prob for unsupervised)
        main_loss = loss_fn(output, None, data)
        reg_loss = model.regularization_loss()
        loss = main_loss + reg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Visualization
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1500)
        
        # Individual flow samples
        realnvp_samples = model.flows[0].sample(1500)
        planar_samples = model.flows[1].sample(1500)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        plot_samples(data, "Original Two Moons", axes[0, 0])
        plot_samples(model_samples, "AMF-VI Samples", axes[0, 1])
        plot_samples(realnvp_samples, "Real-NVP Component", axes[1, 0])
        plot_samples(planar_samples, "Planar Component", axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    train_on_moons()