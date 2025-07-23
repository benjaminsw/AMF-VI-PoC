import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
from amf_vi.flows.iaf import IAFFlow
from data.data_generator import generate_data
import numpy as np
import os
import pickle

class SequentialAMFVI(nn.Module):
    """Sequential training version of AMF-VI with temperature-controlled quality-based weights."""
    
    def __init__(self, dim=2, flow_types=None, use_quality_weights=True, initial_temperature=2.0, final_temperature=0.8):
        super().__init__()
        self.dim = dim
        self.use_quality_weights = use_quality_weights
        
        # Temperature control parameters
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.current_temperature = initial_temperature
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf', 'iaf']
        
        # Create flows
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=4))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=4))
            elif flow_type == 'iaf':
                self.flows.append(IAFFlow(dim, n_layers=4))
        
        # Track if flows are trained
        self.flows_trained = False
        # Store training data for quality computation
        self.training_data = None
    
    def set_temperature(self, temperature):
        """Set the current temperature for weight mixing."""
        self.current_temperature = temperature
    
    def anneal_temperature(self, progress):
        """Anneal temperature based on training progress (0.0 to 1.0)."""
        self.current_temperature = self.initial_temperature * (1 - progress) + self.final_temperature * progress
    
    def compute_quality_scores(self, samples, target_data, method='mmd'):
        """
        Compute per-sample quality scores using MMD or likelihood-based metrics.
        
        Args:
            samples: torch.Tensor [batch_size, dim] - samples to evaluate
            target_data: torch.Tensor [n_target, dim] - reference target data
            method: str - 'mmd', 'knn', or 'likelihood'
        
        Returns:
            torch.Tensor [batch_size] - quality scores (higher = better)
        """
        if method == 'mmd':
            return self._compute_mmd_scores(samples, target_data)
        elif method == 'knn':
            return self._compute_knn_scores(samples, target_data)
        elif method == 'likelihood':
            return self._compute_likelihood_scores(samples, target_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_mmd_scores(self, samples, target_data, bandwidth=1.0):
        """Compute MMD-based quality scores per sample."""
        batch_size = samples.size(0)
        n_target = target_data.size(0)
        
        # Expand for pairwise distance computation
        samples_exp = samples.unsqueeze(1)  # [batch, 1, dim]
        target_exp = target_data.unsqueeze(0)  # [1, n_target, dim]
        
        # Compute pairwise distances to target data
        distances = torch.sum((samples_exp - target_exp) ** 2, dim=2)  # [batch, n_target]
        
        # RBF kernel with target data
        kernel_target = torch.exp(-distances / (2 * bandwidth ** 2))
        
        # Average kernel values as quality score
        quality_scores = kernel_target.mean(dim=1)  # [batch_size]
        
        return quality_scores

    def _compute_knn_scores(self, samples, target_data, k=5):
        """Compute k-NN based quality scores per sample."""
        batch_size = samples.size(0)
        
        # Expand for pairwise distance computation
        samples_exp = samples.unsqueeze(1)  # [batch, 1, dim]
        target_exp = target_data.unsqueeze(0)  # [1, n_target, dim]
        
        # Compute distances to all target points
        distances = torch.sum((samples_exp - target_exp) ** 2, dim=2)  # [batch, n_target]
        
        # Find k nearest neighbors and compute average distance
        knn_distances, _ = torch.topk(distances, k, dim=1, largest=False)
        avg_knn_distance = knn_distances.mean(dim=1)  # [batch_size]
        
        # Convert to quality score (lower distance = higher quality)
        quality_scores = 1.0 / (1.0 + avg_knn_distance)
        
        return quality_scores

    def _compute_likelihood_scores(self, samples, target_data):
        """Compute likelihood-based quality scores using kernel density estimation."""
        batch_size = samples.size(0)
        bandwidth = 0.5  # Fixed bandwidth for simplicity
        
        # Expand for pairwise distance computation
        samples_exp = samples.unsqueeze(1)  # [batch, 1, dim]
        target_exp = target_data.unsqueeze(0)  # [1, n_target, dim]
        
        # Compute Gaussian kernel density estimate
        distances = torch.sum((samples_exp - target_exp) ** 2, dim=2)  # [batch, n_target]
        kernel_values = torch.exp(-distances / (2 * bandwidth ** 2))
        
        # Normalize and compute log-likelihood
        density_estimates = kernel_values.mean(dim=1)  # [batch_size]
        quality_scores = torch.log(density_estimates + 1e-8)
        
        return quality_scores
    
    def compute_flow_quality_weights(self, n_samples=500):
        """Compute temperature-controlled quality-based weights for each flow."""
        if not self.use_quality_weights or self.training_data is None:
            # Return uniform weights
            return torch.ones(len(self.flows), device=next(self.parameters()).device) / len(self.flows)
        
        flow_qualities = []
        
        with torch.no_grad():
            for flow in self.flows:
                flow.eval()
                # Sample from this flow
                samples = flow.sample(n_samples)
                # Compute quality scores
                quality_scores = self.compute_quality_scores(samples, self.training_data, method='mmd')
                # Use mean quality as flow's overall quality
                flow_quality = quality_scores.mean()
                flow_qualities.append(flow_quality)
        
        # Convert to tensor and apply temperature-controlled softmax
        flow_qualities = torch.stack(flow_qualities)
        weights = torch.softmax(flow_qualities / self.current_temperature, dim=0)
        
        return weights
    
    def train_flows_independently(self, data, epochs=1000, lr=1e-4):
        """Stage 1: Train each flow independently with temperature annealing."""
        print("🔄 Stage 1: Training flows independently...")
        print(f"🌡️ Temperature annealing: {self.initial_temperature:.2f} → {self.final_temperature:.2f}")
        
        # Store training data for quality computation
        self.training_data = data.clone()
        
        flow_losses = []
        
        for i, flow in enumerate(self.flows):
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Create optimizer for this flow only
            optimizer = optim.Adam(flow.parameters(), lr=lr)
            losses = []
            
            for epoch in range(epochs):
                # Anneal temperature during training
                progress = epoch / epochs
                self.anneal_temperature(progress)
                
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
                    print(f"    Epoch {epoch}: Loss = {loss.item():.4f}, T = {self.current_temperature:.3f}")
            
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
    
    def forward(self, x):
        """Forward pass with temperature-controlled quality-based or uniform weights."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get flow predictions
        flow_predictions = self.get_flow_predictions(x)
        
        if self.use_quality_weights:
            # Compute temperature-controlled quality-based weights
            flow_weights = self.compute_flow_quality_weights()
            # Expand weights for batch
            batch_size = x.size(0)
            weights = flow_weights.unsqueeze(0).expand(batch_size, -1)
        else:
            # Use uniform weights
            batch_size = x.size(0)
            weights = torch.ones(batch_size, len(self.flows), device=x.device) / len(self.flows)
        
        # Compute mixture log probability
        weighted_log_probs = flow_predictions + torch.log(weights + 1e-8)
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return {
            'log_prob': flow_predictions,
            'weights': weights,
            'mixture_log_prob': mixture_log_prob,
            'temperature': self.current_temperature,
        }
    
    def sample(self, n_samples):
        """Sample from the mixture with temperature-controlled quality-based or uniform weights."""
        device = next(self.parameters()).device
        
        if self.use_quality_weights:
            # Get temperature-controlled quality-based weights
            flow_weights = self.compute_flow_quality_weights()
            
            # Sample proportionally to weights
            all_samples = []
            for i, (flow, weight) in enumerate(zip(self.flows, flow_weights)):
                n_flow_samples = int(n_samples * weight.item())
                if n_flow_samples > 0:
                    flow.eval()
                    with torch.no_grad():
                        samples = flow.sample(n_flow_samples)
                        all_samples.append(samples)
            
            # Handle remaining samples due to rounding
            total_sampled = sum(samples.size(0) for samples in all_samples)
            remaining = n_samples - total_sampled
            if remaining > 0:
                # Add remaining samples from best flow (highest weight)
                best_flow_idx = torch.argmax(flow_weights)
                with torch.no_grad():
                    extra_samples = self.flows[best_flow_idx].sample(remaining)
                    all_samples.append(extra_samples)
            
        else:
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
        
        return torch.cat(all_samples, dim=0) if all_samples else torch.empty(0, self.dim, device=device)

    def evaluate_sample_quality(self, samples, target_data):
        """Evaluate quality of generated samples."""
        with torch.no_grad():
            quality_scores = self.compute_quality_scores(samples, target_data, method='mmd')
            return quality_scores

def train_sequential_amf_vi(dataset_name='multimodal', show_plots=True, save_plots=False, use_quality_weights=True, initial_temp=2.0, final_temp=0.8):
    """Train sequential AMF-VI with temperature-controlled quality-based or uniform weights."""
    
    weight_type = f"Quality-Based (T={initial_temp:.1f}→{final_temp:.1f})" if use_quality_weights else "Uniform"
    print(f"🚀 Sequential AMF-VI Experiment ({weight_type} Weights) on {dataset_name}")
    print("=" * 60)
    
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Create sequential model with temperature control
    model = SequentialAMFVI(
        dim=2, 
        flow_types=['realnvp', 'maf', 'iaf'], 
        use_quality_weights=use_quality_weights,
        initial_temperature=initial_temp,
        final_temperature=final_temp
    )
    model = model.to(device)
    
    # Stage 1: Train flows independently
    flow_losses = model.evi_enhanced_flow_training(data, epochs=200, lr=1e-3, tau_evi=0.01)
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000, use_boundary_escape=True)
        
        # Get flow weights for analysis
        if use_quality_weights:
            flow_weights = model.compute_flow_quality_weights()
            print(f"\n🎯 Flow Quality Weights (T={model.current_temperature:.3f}): {flow_weights.cpu().numpy()}")
        
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
        
        # Plot sequential model samples
        model_np = model_samples.cpu().numpy()
        axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
        title_suffix = f" (T-Controlled)" if use_quality_weights else " (Uniform)"
        axes[0, 1].set_title(f'Sequential AMF-VI Samples{title_suffix}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot individual flows
        colors = ['green', 'orange', 'purple']
        for i, (name, samples) in enumerate(flow_samples.items()):
            if i < 3:
                row, col = (0, 2) if i == 0 else (1, i-1)
                samples_np = samples.cpu().numpy()
                axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                     alpha=0.6, c=colors[i], s=20)
                flow_title = f'{name.upper()} Flow'
                if use_quality_weights and i < len(flow_weights):
                    flow_title += f' (w={flow_weights[i]:.3f})'
                axes[row, col].set_title(flow_title)
                axes[row, col].grid(True, alpha=0.3)
        
        # Plot training losses (flow losses only)
        if flow_losses:
            axes[1, 2].plot(flow_losses[0], label='Real-NVP', color='green', linewidth=2, alpha=0.7)
            axes[1, 2].plot(flow_losses[1], label='MAF', color='orange', linewidth=2, alpha=0.7)
            axes[1, 2].plot(flow_losses[2], label='IAF', color='purple', linewidth=2, alpha=0.7)
        axes[1, 2].set_title('Individual Flow Training Losses')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.suptitle(f'Sequential AMF-VI Results ({weight_type}) - {dataset_name.title()}', fontsize=16)
        
        # Save plot if requested
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            suffix = f"temp_{initial_temp}_{final_temp}" if use_quality_weights else "uniform"
            plot_path = os.path.join(results_dir, f'sequential_amf_vi_results_{dataset_name}_{suffix}.png')
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
        print(f"Final temperature: {model.current_temperature:.3f}")
        
        # Check flow diversity
        print("\n🔍 Flow Specialization Analysis:")
        for i, (name, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            quality_info = ""
            if use_quality_weights and i < len(flow_weights):
                quality_info = f" (Quality Weight: {flow_weights[i]:.3f})"
            print(f"{name.upper()}: Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]{quality_info}")
    
    # Save trained model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    suffix = f"temp_{initial_temp}_{final_temp}" if use_quality_weights else "uniform"
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}_{suffix}.pkl')
    
    model_data = {
        'model': model, 
        'losses': flow_losses, 
        'dataset': dataset_name, 
        'use_quality_weights': use_quality_weights,
        'initial_temperature': initial_temp,
        'final_temperature': final_temp
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses

if __name__ == "__main__":
    # Run the sequential AMF-VI experiment on multiple datasets
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Train with temperature-controlled quality-based weights
        model_temp, flow_losses_temp = train_sequential_amf_vi(
            dataset_name, 
            show_plots=False, 
            save_plots=True,
            use_quality_weights=True,
            initial_temp=2.0,
            final_temp=0.8
        )
        
        # Train with uniform weights for comparison
        model_uniform, flow_losses_uniform = train_sequential_amf_vi(
            dataset_name, 
            show_plots=False, 
            save_plots=True,
            use_quality_weights=False
        )