import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.planar import PlanarFlow
from amf_vi.flows.radial import RadialFlow
from data.data_generator import generate_data
import numpy as np

class SequentialAMFVI(nn.Module):
    """Sequential training version of AMF-VI for experimental comparison."""
    
    def __init__(self, dim=2, flow_types=None):
        super().__init__()
        self.dim = dim
        
        if flow_types is None:
            flow_types = ['realnvp', 'planar', 'radial']
        
        # Create flows
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=4))
            elif flow_type == 'planar':
                self.flows.append(PlanarFlow(dim, n_layers=8))
            elif flow_type == 'radial':
                self.flows.append(RadialFlow(dim, n_layers=8))
        
        # Meta-learner (posterior prediction network)
        self.meta_learner = nn.Sequential(
            nn.Linear(dim + len(self.flows), 64),  # input + flow predictions
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.flows)),
            nn.Softmax(dim=1)
        )
        
        # Track if flows are trained
        self.flows_trained = False
    
    def train_flows_independently(self, data, epochs=200, lr=1e-3):
        """Stage 1: Train each flow independently."""
        print("🔄 Stage 1: Training flows independently...")
        
        flow_losses = []
        device = data.device
        
        for i, flow in enumerate(self.flows):
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Create optimizer for this flow only
            optimizer = optim.Adam(flow.parameters(), lr=lr)
            losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Individual flow loss (negative log-likelihood)
                log_prob = flow.log_prob(data)
                loss = -log_prob.mean()
                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 50 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
            
            flow_losses.append(losses)
            print(f"    Final loss: {losses[-1]:.4f}")
        
        self.flows_trained = True
        return flow_losses
    
    def get_flow_predictions(self, x):
        """Get predictions from all pre-trained flows."""
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        flow_log_probs = []
        for flow in self.flows:
            flow.eval()
            with torch.no_grad():
                log_prob = flow.log_prob(x)
                flow_log_probs.append(log_prob.unsqueeze(1))
        
        return torch.cat(flow_log_probs, dim=1)  # [batch, n_flows]
    
    def train_meta_learner(self, data, epochs=300, lr=1e-3):
        """Stage 2: Train meta-learner to combine flows."""
        print("🔄 Stage 2: Training meta-learner...")
        
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        # Get flow predictions (fixed)
        flow_predictions = self.get_flow_predictions(data)
        
        # Create input for meta-learner: [x, flow_predictions]
        meta_input = torch.cat([data, flow_predictions], dim=1)
        
        # Train meta-learner
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Get adaptive weights from meta-learner
            weights = self.meta_learner(meta_input)
            
            # Compute weighted mixture log probability
            weighted_log_probs = flow_predictions + torch.log(weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            
            # Meta-learner loss (negative log-likelihood)
            loss = -mixture_log_prob.mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        print(f"  Final loss: {losses[-1]:.4f}")
        return losses
    
    def forward(self, x):
        """Forward pass through the sequential model."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get flow predictions
        flow_predictions = self.get_flow_predictions(x)
        
        # Get meta-learner weights
        meta_input = torch.cat([x, flow_predictions], dim=1)
        weights = self.meta_learner(meta_input)
        
        return {
            'log_prob': flow_predictions,
            'weights': weights,
        }
    
    def sample(self, n_samples):
        """Sample from the mixture (uniform weights for simplicity)."""
        device = next(self.parameters()).device
        
        # Sample uniformly from flows
        samples_per_flow = n_samples // len(self.flows)
        all_samples = []
        
        for flow in self.flows:
            flow.eval()
            with torch.no_grad():
                samples = flow.sample(samples_per_flow)
                all_samples.append(samples)
        
        # Add remaining samples from first flow
        remaining = n_samples - len(self.flows) * samples_per_flow
        if remaining > 0:
            with torch.no_grad():
                extra_samples = self.flows[0].sample(remaining)
                all_samples.append(extra_samples)
        
        return torch.cat(all_samples, dim=0)

def train_sequential_amf_vi(dataset_name='multimodal', show_plots=True):
    """Train sequential AMF-VI and compare with current approach."""
    
    print(f"🚀 Sequential AMF-VI Experiment on {dataset_name}")
    print("=" * 60)
    
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Create sequential model
    model = SequentialAMFVI(dim=2, flow_types=['realnvp', 'planar', 'radial'])
    model = model.to(device)
    
    # Stage 1: Train flows independently
    flow_losses = model.train_flows_independently(data, epochs=200, lr=1e-3)
    
    # Stage 2: Train meta-learner
    meta_losses = model.train_meta_learner(data, epochs=300, lr=1e-3)
    
    # Evaluation and visualization
    if show_plots:
        print("\n🎨 Generating visualizations...")
        
        model.eval()
        with torch.no_grad():
            # Generate samples
            model_samples = model.sample(1000)
            
            # Individual flow samples
            flow_samples = {}
            flow_names = ['realnvp', 'planar', 'radial']
            for i, name in enumerate(flow_names):
                flow_samples[name] = model.flows[i].sample(1000)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot target data
            data_np = data.cpu().numpy()
            axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
            axes[0, 0].set_title('Target Data')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot sequential model samples
            model_np = model_samples.cpu().numpy()
            axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
            axes[0, 1].set_title('Sequential AMF-VI Samples')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot individual flows
            colors = ['green', 'orange', 'purple']
            for i, (name, samples) in enumerate(flow_samples.items()):
                if i < 3:
                    row, col = (0, 2) if i == 0 else (1, i-1)
                    samples_np = samples.cpu().numpy()
                    axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                         alpha=0.6, c=colors[i], s=20)
                    axes[row, col].set_title(f'{name.title()} Flow')
                    axes[row, col].grid(True, alpha=0.3)
            
            # Plot training losses
            axes[1, 2].plot(meta_losses, label='Meta-learner', color='red', linewidth=2)
            axes[1, 2].set_title('Meta-learner Training Loss')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.suptitle(f'Sequential AMF-VI Results - {dataset_name.title()}', fontsize=16)
            plt.show()
            
            # Print analysis
            print("\n📊 Analysis:")
            print(f"Target data mean: {data.mean(dim=0).cpu().numpy()}")
            print(f"Sequential model mean: {model_samples.mean(dim=0).cpu().numpy()}")
            print(f"Target data std: {data.std(dim=0).cpu().numpy()}")
            print(f"Sequential model std: {model_samples.std(dim=0).cpu().numpy()}")
            
            # Check flow diversity
            print("\n🔍 Flow Specialization Analysis:")
            for i, (name, samples) in enumerate(flow_samples.items()):
                mean = samples.mean(dim=0).cpu().numpy()
                std = samples.std(dim=0).cpu().numpy()
                print(f"{name.capitalize()}: Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
    
    return model, flow_losses, meta_losses

def compare_with_current_approach():
    """Compare sequential vs current approach."""
    print("🔬 Experimental Comparison: Sequential vs Current AMF-VI")
    print("=" * 70)
    
    # Test sequential approach
    seq_model, flow_losses, meta_losses = train_sequential_amf_vi('multimodal', show_plots=True)
    
    print("\n" + "=" * 70)
    print("Expected Issues with Sequential Training:")
    print("1. All flows likely learned similar distributions (mode collapse)")
    print("2. Meta-learner has to work with redundant flows")
    print("3. Less effective multimodal coverage")
    print("4. No adaptive specialization during flow training")
    
    return seq_model

if __name__ == "__main__":
    # Run the experimental comparison
    model = compare_with_current_approach()