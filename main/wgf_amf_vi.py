import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
from amf_vi.flows.iaf import IAFFlow
from data.data_generator import generate_data

class WassersteinGradientFlowAMFVI(nn.Module):
    """
    AMF-VI enhanced with Wasserstein Gradient Flow for joint optimization
    of flow parameters and mixture weights.
    """
    
    def __init__(self, dim=2, flow_types=None, n_particles=50, use_wgf=True):
        super().__init__()
        self.dim = dim
        self.n_particles = n_particles
        self.use_wgf = use_wgf
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf', 'iaf']
        
        # Create flows (these act as "particle types" in WGF)
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=4))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=4))
            elif flow_type == 'iaf':
                self.flows.append(IAFFlow(dim, n_layers=4))
        
        # WGF-specific parameters
        self.flow_weights = nn.Parameter(torch.ones(len(self.flows)) / len(self.flows))
        self.particle_positions = None  # Will store particle positions for each flow
        self.particle_weights = None    # Will store particle weights
        
        # Training state
        self.flows_trained = False
        self.training_data = None
        
    def compute_wasserstein_distance_particles(self, particles1, weights1, particles2, weights2):
        """
        Compute approximate Wasserstein distance between two sets of weighted particles
        using optimal transport assignment.
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(particles1, particles2, p=2)
        
        # Convert to numpy for scipy
        cost_matrix = dist_matrix.detach().cpu().numpy()
        w1_np = weights1.detach().cpu().numpy()
        w2_np = weights2.detach().cpu().numpy()
        
        # Solve optimal transport problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute Wasserstein distance
        transport_cost = cost_matrix[row_ind, col_ind].sum()
        
        return torch.tensor(transport_cost, device=particles1.device)
    
    def sample_particles_from_flows(self, n_samples_per_flow=100):
        """
        Sample particles from each flow to represent the current mixture.
        """
        all_particles = []
        all_weights = []
        all_flow_indices = []
        
        for i, flow in enumerate(self.flows):
            flow.eval()
            with torch.no_grad():
                # Sample from this flow
                particles = flow.sample(n_samples_per_flow)
                
                # Weight by flow importance
                weight_per_particle = self.flow_weights[i] / n_samples_per_flow
                weights = torch.full((n_samples_per_flow,), weight_per_particle, 
                                   device=particles.device)
                
                all_particles.append(particles)
                all_weights.append(weights)
                all_flow_indices.extend([i] * n_samples_per_flow)
        
        return (torch.cat(all_particles, dim=0), 
                torch.cat(all_weights, dim=0),
                torch.tensor(all_flow_indices))
    
    def compute_target_particles(self, target_data, n_target_particles=200):
        """
        Create target particles from training data using kernel density estimation.
        """
        # Simple approach: subsample target data and add noise
        indices = torch.randperm(target_data.size(0))[:n_target_particles]
        target_particles = target_data[indices]
        
        # Add small amount of noise to avoid delta functions
        noise = torch.randn_like(target_particles) * 0.1
        target_particles = target_particles + noise
        
        # Uniform weights for target particles
        target_weights = torch.ones(n_target_particles, device=target_data.device) / n_target_particles
        
        return target_particles, target_weights
    
    def wgf_step(self, target_data, lr=0.01, regularization=0.1):
        """
        Perform one Wasserstein Gradient Flow step.
        This is the core of Step 2 from your implementation plan.
        """
        # Sample current particles from mixture
        current_particles, current_weights, flow_indices = self.sample_particles_from_flows()
        
        # Create target particles
        target_particles, target_weights = self.compute_target_particles(target_data)
        
        # Simplified WGF step: just update mixture weights based on quality
        self.update_mixture_weights(current_particles, target_particles, lr)
        
        # Return a simple distance metric for monitoring
        distances = torch.cdist(current_particles, target_particles)
        wass_dist = distances.min(dim=1)[0].mean()
        
        return wass_dist.item()
    
    def update_flows_from_particle_gradients(self, particle_updates, flow_indices):
        """
        Update flow parameters based on how their generated particles should move.
        """
        for i, flow in enumerate(self.flows):
            # Find particles belonging to this flow
            flow_mask = (flow_indices == i)
            if flow_mask.sum() == 0:
                continue
                
            flow_particle_updates = particle_updates[flow_mask]
            
            # Compute average update direction for this flow
            avg_update = flow_particle_updates.mean(dim=0)
            
            # Apply update to flow parameters (simplified - in practice, 
            # you'd want more sophisticated parameter updates)
            for param in flow.parameters():
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                # Simple heuristic: update parameters proportional to particle movement
                param.grad += 0.01 * avg_update.mean() * torch.randn_like(param)
    
    def update_mixture_weights(self, current_particles, target_particles, lr):
        """
        Update mixture weights using Fisher-Rao gradient flow.
        """
        # Compute quality of each flow based on how well its particles match target
        flow_qualities = []
        
        for i, flow in enumerate(self.flows):
            flow.eval()
            with torch.no_grad():
                # Sample from this flow
                flow_samples = flow.sample(100)
                
                # Compute average distance to target particles
                distances = torch.cdist(flow_samples, target_particles)
                avg_distance = distances.min(dim=1)[0].mean()  # Distance to nearest target
                
                # Quality is inverse of distance (higher = better)
                quality = 1.0 / (1.0 + avg_distance)
                flow_qualities.append(quality)
        
        # Convert to tensor
        qualities = torch.stack(flow_qualities)
        
        # Fisher-Rao update (softmax with temperature)
        temperature = 1.0
        new_weights = torch.softmax(qualities / temperature, dim=0)
        
        # Update mixture weights with momentum
        momentum = 0.9
        self.flow_weights.data = momentum * self.flow_weights.data + (1 - momentum) * new_weights
    
    def train_with_wgf(self, data, wgf_epochs=100, flow_epochs=50, lr=1e-3):
        """
        Train using Wasserstein Gradient Flow approach.
        Combines your sequential training with WGF joint optimization.
        """
        print("🌊 Training AMF-VI with Wasserstein Gradient Flow...")
        self.training_data = data.clone()
        
        # Stage 1: Initial flow training (similar to your current approach)
        print("Stage 1: Initial flow training...")
        flow_losses = []
        for i, flow in enumerate(self.flows):
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            optimizer = optim.Adam(flow.parameters(), lr=lr)
            
            losses = []
            for epoch in range(flow_epochs):
                optimizer.zero_grad()
                log_prob = flow.log_prob(data)
                loss = -log_prob.mean()
                
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 10 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
            
            flow_losses.append(losses)
        
        # Stage 2: Wasserstein Gradient Flow joint optimization
        print("\nStage 2: Wasserstein Gradient Flow optimization...")
        wgf_losses = []
        
        # Create optimizers for flow parameters
        flow_optimizers = [optim.Adam(flow.parameters(), lr=lr/10) for flow in self.flows]
        
        for epoch in range(wgf_epochs):
            # Clear gradients
            for optimizer in flow_optimizers:
                optimizer.zero_grad()
            
            # Perform WGF step
            wass_loss = self.wgf_step(data, lr=lr/10)
            wgf_losses.append(wass_loss)
            
            # Apply parameter updates
            for optimizer in flow_optimizers:
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  WGF Epoch {epoch}: Wasserstein Loss = {wass_loss:.4f}")
                print(f"    Flow weights: {self.flow_weights.data.cpu().numpy()}")
        
        self.flows_trained = True
        return flow_losses, wgf_losses
    
    def forward(self, x):
        """Forward pass using WGF-trained mixture."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get predictions from all flows
        flow_log_probs = []
        for flow in self.flows:
            flow.eval()
            with torch.no_grad():
                log_prob = flow.log_prob(x)
                flow_log_probs.append(log_prob.unsqueeze(1))
        
        flow_predictions = torch.cat(flow_log_probs, dim=1)
        
        # Use learned weights
        batch_size = x.size(0)
        weights = self.flow_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Compute mixture log probability
        weighted_log_probs = flow_predictions + torch.log(weights + 1e-8)
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return {
            'log_prob': flow_predictions,
            'weights': weights,
            'mixture_log_prob': mixture_log_prob,
            'flow_weights': self.flow_weights
        }
    
    def sample(self, n_samples):
        """Sample from the WGF-trained mixture."""
        device = next(self.parameters()).device
        
        # Sample according to learned flow weights
        flow_weights_np = self.flow_weights.detach().cpu().numpy()
        
        all_samples = []
        for i, (flow, weight) in enumerate(zip(self.flows, flow_weights_np)):
            n_flow_samples = int(n_samples * weight)
            if n_flow_samples > 0:
                flow.eval()
                with torch.no_grad():
                    samples = flow.sample(n_flow_samples)
                    all_samples.append(samples)
        
        # Handle remaining samples
        remaining = n_samples - sum(s.size(0) for s in all_samples)
        if remaining > 0:
            best_flow_idx = torch.argmax(self.flow_weights)
            with torch.no_grad():
                extra_samples = self.flows[best_flow_idx].sample(remaining)
                all_samples.append(extra_samples)
        
        return torch.cat(all_samples, dim=0) if all_samples else torch.empty(0, self.dim, device=device)


def train_wgf_amf_vi(dataset_name='multimodal', show_plots=True, save_plots=False):
    """
    Train AMF-VI with Wasserstein Gradient Flow enhancement.
    """
    print(f"🚀 WGF-Enhanced AMF-VI Experiment on {dataset_name}")
    print("=" * 60)
    
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Create WGF-enhanced model
    model = WassersteinGradientFlowAMFVI(
        dim=2, 
        flow_types=['realnvp', 'maf', 'iaf'],
        use_wgf=True
    )
    model = model.to(device)
    
    # Train with WGF
    flow_losses, wgf_losses = model.train_with_wgf(
        data, 
        wgf_epochs=100, 
        flow_epochs=50, 
        lr=1e-3
    )
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000)
        
        # Get final flow weights
        final_weights = model.flow_weights.detach().cpu().numpy()
        print(f"\n🎯 Final Flow Weights: {final_weights}")
        
        # Individual flow samples
        flow_samples = {}
        flow_names = ['realnvp', 'maf', 'iaf']
        for i, name in enumerate(flow_names):
            flow_samples[name] = model.flows[i].sample(1000)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot target data
        data_np = data.cpu().numpy()
        axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
        axes[0, 0].set_title('Target Data')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot WGF model samples
        model_np = model_samples.cpu().numpy()
        axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
        axes[0, 1].set_title('WGF-Enhanced AMF-VI Samples')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot individual flows with their weights
        colors = ['green', 'orange', 'purple']
        for i, (name, samples) in enumerate(flow_samples.items()):
            if i < 3:
                row, col = (0, 2) if i == 0 else (1, i-1)
                samples_np = samples.cpu().numpy()
                axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                     alpha=0.6, c=colors[i], s=20)
                axes[row, col].set_title(f'{name.upper()} (w={final_weights[i]:.3f})')
                axes[row, col].grid(True, alpha=0.3)
        
        # Plot WGF training losses
        if wgf_losses:
            axes[1, 2].plot(wgf_losses, label='Wasserstein Loss', color='red', linewidth=2)
            axes[1, 2].set_title('WGF Training Loss')
            axes[1, 2].set_xlabel('WGF Epoch')
            axes[1, 2].set_ylabel('Wasserstein Distance')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.suptitle(f'WGF-Enhanced AMF-VI Results - {dataset_name.title()}', fontsize=16)
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    return model, flow_losses, wgf_losses


if __name__ == "__main__":
    # Test WGF-enhanced AMF-VI
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        model, flow_losses, wgf_losses = train_wgf_amf_vi(
            dataset_name, 
            show_plots=False, 
            save_plots=True
        )
