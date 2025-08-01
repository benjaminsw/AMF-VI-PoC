import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
from amf_vi.flows.iaf import IAFFlow
from data.data_generator import generate_data
import numpy as np
import os
import pickle

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

class SequentialAMFVI(nn.Module):
    """Sequential training version of AMF-VI with learnable weights."""
    
    def __init__(self, dim=2, flow_types=None, weight_update_method='moving_average'):
        super().__init__()
        self.dim = dim
        # Store dataset name for fresh data generation during weight learning
        self.dataset_name = dataset_name 
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf']
        
        # Create flows
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=8))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=8))
            elif flow_type == 'iaf':
                self.flows.append(IAFFlow(dim, n_layers=1))
        
        # Initialize weights as parameters (not log_weights for moving average)
        self.weights = nn.Parameter(torch.ones(len(self.flows)) / len(self.flows))
        self.weight_update_method = weight_update_method
        
        # Moving average parameters
        self.alpha = 0.9  # Moving average decay factor
        self.weight_lr = 0.01
        self.weight_history = []
        
        # Track if flows are trained
        self.flows_trained = False
        self.weights_trained = False
    
    def train_flows_independently(self, data, epochs=1000, lr=1e-4):
        """Stage 1: Train each flow independently."""
        print("🔄 Stage 1: Training flows independently...")
        
        flow_losses = []
        
        for i, flow in enumerate(self.flows):
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Create optimizer for this flow only
            optimizer = optim.Adam(flow.parameters(), lr=lr)
            losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                try:
                    # Individual flow loss (negative log-likelihood)
                    log_prob = flow.log_prob(data)
                    loss = -log_prob.mean()
                    
                    # Check if loss requires grad
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                    else:
                        print(f"    Warning: Loss doesn't require grad at epoch {epoch}")
                    
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        print(f"    Skipping gradient step at epoch {epoch}: {e}")
                        # Create a dummy loss for this step
                        dummy_loss = torch.tensor(float('nan'), requires_grad=True)
                        losses.append(dummy_loss.item())
                        continue
                    else:
                        raise e
                
                losses.append(loss.item())
                
                if epoch % 50 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
            
            flow_losses.append(losses)
            print(f"    Final loss: {losses[-1]:.4f}")
        
        self.flows_trained = True
        return flow_losses
    
    def train_mixture_weights_moving_average(self, data, epochs=500):
        """Stage 2: Learn weights using Moving Average of Likelihoods."""
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        print("🔄 Stage 2: Learning mixture weights (Moving Average)...")
        
        weight_losses = []
        
        for epoch in range(epochs):
            # Generate fresh data for each epoch to avoid bias
            dataset_name = getattr(self, 'dataset_name', 'multimodal')  # Default fallback
            fresh_data = generate_data(dataset_name, n_samples=1000)
            fresh_data = fresh_data.to(data.device)
            
            # Get flow log probabilities on fresh data
            flow_log_probs = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    log_prob = flow.log_prob(fresh_data)
                    flow_log_probs.append(log_prob.mean().item())  # Average over batch
            
            # Convert to likelihoods and normalize (softmax)
            flow_log_probs_tensor = torch.tensor(flow_log_probs)
            normalized_likelihoods = F.softmax(flow_log_probs_tensor, dim=0)
            
            # Moving average update: weight_i = α * old_weight_i + (1-α) * normalized_likelihood_i
            with torch.no_grad():
                old_weights = self.weights.data.clone()
                new_weights = self.alpha * old_weights + (1 - self.alpha) * normalized_likelihoods
                self.weights.data = new_weights
            
            # Compute mixture log probability for loss tracking (use original data for consistency)
            batch_weights = self.weights.unsqueeze(0).expand(data.size(0), -1)
            flow_predictions = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    log_prob = flow.log_prob(data)
                    flow_predictions.append(log_prob.unsqueeze(1))
            
            flow_predictions = torch.cat(flow_predictions, dim=1)
            weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            loss = -mixture_log_prob.mean()
            
            weight_losses.append(loss.item())
            
            if epoch % 100 == 0:
                current_weights = self.weights.detach().cpu().numpy()
                print(f"    Epoch {epoch}: Loss = {loss.item():.4f}, Weights = {current_weights}")
        
        final_weights = self.weights.detach().cpu().numpy()
        print(f"    Final weights: {final_weights}")
        
        self.weights_trained = True
        self.weight_history = weight_losses
        return weight_losses
    
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
    
    def forward(self, x):
        """Forward pass with learned or uniform weights."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get flow predictions
        flow_predictions = self.get_flow_predictions(x)
        
        # Use learned weights if available, otherwise uniform
        if self.weights_trained:
            weights = self.weights
        else:
            weights = torch.ones(len(self.flows), device=x.device) / len(self.flows)
        
        batch_size = x.size(0)
        batch_weights = weights.unsqueeze(0).expand(batch_size, -1)
        
        # Compute mixture log probability
        weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return {
            'log_prob': flow_predictions,
            'weights': batch_weights,
            'mixture_log_prob': mixture_log_prob,
        }
    
    def log_prob(self, x):
        """Compute log probability of data under the mixture model."""
        return self.forward(x)['mixture_log_prob']
    
    def sample(self, n_samples):
        """Sample from the mixture with learned or uniform weights."""
        device = next(self.parameters()).device
        
        # Get current weights
        if self.weights_trained:
            weights = self.weights.detach().cpu().numpy()
        else:
            weights = np.ones(len(self.flows)) / len(self.flows)
        
        # Sample according to learned weights
        flow_indices = np.random.choice(len(self.flows), size=n_samples, p=weights)
        unique_indices, counts = np.unique(flow_indices, return_counts=True)
        
        all_samples = []
        
        for idx, count in zip(unique_indices, counts):
            flow = self.flows[idx]
            flow.eval()
            with torch.no_grad():
                samples = flow.sample(count)
                all_samples.append(samples)
        
        return torch.cat(all_samples, dim=0)

def train_sequential_amf_vi(dataset_name='multimodal', show_plots=True, save_plots=False):
    """Train sequential AMF-VI with learnable weights."""
    
    print(f"🚀 Sequential AMF-VI Experiment on {dataset_name}")
    print("=" * 60)
    
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Create sequential model with moving average weight updates
    model = SequentialAMFVI(dim=2, flow_types=['realnvp', 'maf'], weight_update_method='moving_average')
    model.dataset_name = dataset_name  # Set dataset name for fresh data generation
    model = model.to(device)
    
    # Stage 1: Train flows independently
    flow_losses = model.train_flows_independently(data, epochs=500, lr=1e-3)
    
    # Stage 2: Learn mixture weights using moving average (simplified call)
    weight_losses = model.train_mixture_weights_moving_average(data, epochs=500)
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000)
        
        # Individual flow samples
        flow_samples = {}
        flow_names = ['realnvp', 'maf']
        for i, name in enumerate(flow_names):
            flow_samples[name] = model.flows[i].sample(1000)
        
        # Create visualization
        # Change subplot creation from 2x2 to 2x3
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        # Plot target data (unchanged)
        data_np = data.cpu().numpy()
        axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
        axes[0, 0].set_title('Target Data')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot sequential model samples (unchanged)
        model_np = model_samples.cpu().numpy()
        axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
        axes[0, 1].set_title('Sequential AMF-VI Samples (Moving Avg Weights)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot individual flows (unchanged - still uses bottom row positions 0,1)
        colors = ['green', 'orange']
        for i, (name, samples) in enumerate(flow_samples.items()):
            row, col = (1, i)
            samples_np = samples.cpu().numpy()
            axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                alpha=0.6, c=colors[i], s=20)
            axes[row, col].set_title(f'{name.upper()} Flow')
            axes[row, col].grid(True, alpha=0.3)

        
        #####################################################################
        # Plot training losses - move to bottom-right (axes[1,2])
        #if flow_losses:
            # Plot flow losses with reduced alpha
            #axes[1, 2].plot(flow_losses[0], label='Real-NVP', color='green', linewidth=1, alpha=0.7)
            #axes[1, 2].plot(flow_losses[1], label='MAF', color='orange', linewidth=1, alpha=0.7)

        # Plot weight learning loss with different scale
        #if weight_losses:
            #ax2 = axes[1, 2].twinx()
            #ax2.plot(weight_losses, label='Weight Learning', color='red', linewidth=2)
            #ax2.set_ylabel('Weight Loss', color='red')
            #ax2.tick_params(axis='y', labelcolor='red')

        #axes[1, 2].set_title('Training Losses')
        #axes[1, 2].set_xlabel('Epoch')
        #axes[1, 2].set_ylabel('Flow Loss')
        #axes[1, 2].grid(True, alpha=0.3)
        #axes[1, 2].legend(loc='upper left')
        #####################################################################
        
        # Plot training losses - move to bottom-right (axes[1,2])
        if flow_losses:
            # Plot flow losses with reduced alpha
            axes[1, 2].plot(flow_losses[0], label='Real-NVP', color='green', linewidth=1, alpha=0.7)
            axes[1, 2].plot(flow_losses[1], label='MAF', color='orange', linewidth=1, alpha=0.7)

        # Plot weight learning loss on SAME scale (remove twinx())
        if weight_losses:
            axes[1, 2].plot(weight_losses, label='Weight Learning', color='red', linewidth=2)

        axes[1, 2].set_title('Training Losses')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')  # Generic label since all on same scale
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()  # Remove loc parameter to let matplotlib choose best location
        
        

        # Hide unused subplot (top-right)
        axes[0, 2].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'Sequential AMF-VI Results (Moving Avg) - {dataset_name.title()}', fontsize=16)
        
        # Save plot if requested
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'sequential_amf_vi_results_{dataset_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {plot_path}")
        
        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # Print analysis
        print("\n📊 Analysis:")
        print(f"Target data mean: {data.mean(dim=0).cpu().numpy()}")
        print(f"Sequential model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        print(f"Target data std: {data.std(dim=0).cpu().numpy()}")
        print(f"Sequential model std: {model_samples.std(dim=0).cpu().numpy()}")
        
        # Check flow diversity and learned weights
        print("\n🔍 Flow Specialization Analysis:")
        learned_weights = model.weights.detach().cpu().numpy()
        for i, (name, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            weight = learned_weights[i]
            print(f"{name.upper()}: Weight={weight:.3f}, Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
        
        # Model complexity analysis
        print("\n🏗️ Model Architecture:")
        total_params = 0
        for i, flow in enumerate(model.flows):
            n_params = sum(p.numel() for p in flow.parameters())
            total_params += n_params
            print(f"{flow_names[i].upper()}: {n_params:,} parameters")
        print(f"Total parameters: {total_params:,}")
        print(f"Weight parameters: {model.weights.numel()}")
    
    # Save trained model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'flow_losses': flow_losses, 
            'weight_losses': weight_losses,
            'dataset': dataset_name
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, weight_losses

if __name__ == "__main__":
    # Run the sequential AMF-VI experiment on multiple datasets
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        model, flow_losses, weight_losses = train_sequential_amf_vi(
            dataset_name, 
            show_plots=False, 
            save_plots=True
        )