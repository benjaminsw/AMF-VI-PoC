import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
from amf_vi.flows.iaf import IAFFlow
from amf_vi.flows.spline import SplineFlow
from data.data_generator import generate_data
import numpy as np
import os
import pickle

class SequentialAMFVI(nn.Module):
    """Sequential training version of AMF-VI with quality-aware meta-learner weights."""
    
    def __init__(self, dim=2, flow_types=None, temperature=1.0, quality_weight=0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature  # Temperature for softmax weighting
        self.quality_weight = quality_weight  # Weight for quality-based reweighting
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf', 'iaf']
        
        # Create flows
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=1))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=1))
            elif flow_type == 'iaf':
                self.flows.append(IAFFlow(dim, n_layers=1))

        # Meta-learner for adaptive weights with quality awareness
        self.meta_learner = nn.Sequential(
            nn.Linear(dim + len(self.flows) + 1, 64),  # +1 for quality score
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.flows)),
            nn.Softmax(dim=1)
        )
        
        # Track training stages
        self.flows_trained = False
        self.meta_trained = False
        
        # Store flow quality scores for dynamic reweighting
        self.flow_quality_scores = torch.ones(len(self.flows))
    
    def compute_quality_scores(self, data, per_sample=False):
        """
        Compute quality scores for each flow based on data fit.
        
        Args:
            data: Input data tensor
            per_sample: If True, return per-sample quality scores
        
        Returns:
            quality_scores: Tensor of shape [n_flows] or [batch_size, n_flows]
        """
        if not self.flows_trained:
            if per_sample:
                return torch.ones(data.size(0), len(self.flows), device=data.device)
            return torch.ones(len(self.flows), device=data.device)
        
        quality_scores = []
        
        for flow in self.flows:
            flow.eval()
            with torch.no_grad():
                log_prob = flow.log_prob(data)
                
                if per_sample:
                    # Per-sample quality: use log probability directly
                    quality = log_prob
                else:
                    # Global quality: use mean log probability
                    quality = log_prob.mean()
                
                quality_scores.append(quality.unsqueeze(-1) if per_sample else quality)
        
        if per_sample:
            quality_scores = torch.cat(quality_scores, dim=1)  # [batch_size, n_flows]
        else:
            quality_scores = torch.stack(quality_scores)  # [n_flows]
        
        # Normalize quality scores using temperature-scaled softmax
        quality_scores = torch.softmax(quality_scores / self.temperature, dim=-1)
        
        return quality_scores
    
    def update_flow_quality_scores(self, data):
        """Update global flow quality scores based on current data."""
        if self.flows_trained:
            self.flow_quality_scores = self.compute_quality_scores(data, per_sample=False)
            print(f"📊 Updated flow quality scores: {[f'{score:.3f}' for score in self.flow_quality_scores.cpu().numpy()]}")
    
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
        
        # Update quality scores after training
        self.update_flow_quality_scores(data)
        
        return flow_losses
    
    def train_meta_learner(self, data, epochs=300, lr=1e-3):
        """Stage 2: Train meta-learner to learn adaptive weights with quality awareness."""
        print("🔄 Stage 2: Training quality-aware meta-learner...")
        
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        # Freeze all flow parameters
        for flow in self.flows:
            for param in flow.parameters():
                param.requires_grad = False
        
        # Get flow predictions and quality scores
        flow_predictions = self.get_flow_predictions(data)
        per_sample_quality = self.compute_quality_scores(data, per_sample=True)
        
        # Average quality score per sample (used as additional input)
        avg_quality_per_sample = per_sample_quality.mean(dim=1, keepdim=True)
        
        # Create input for meta-learner: [x, flow_predictions, avg_quality]
        meta_input = torch.cat([data, flow_predictions, avg_quality_per_sample], dim=1)
        
        # Train meta-learner only
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Get adaptive weights from meta-learner
            weights = self.meta_learner(meta_input)
            
            # Apply quality-aware reweighting
            quality_weighted = weights * per_sample_quality
            normalized_weights = quality_weighted / (quality_weighted.sum(dim=1, keepdim=True) + 1e-8)
            
            # Compute weighted mixture log probability
            weighted_log_probs = flow_predictions + torch.log(normalized_weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            
            # Meta-learner loss with quality regularization
            base_loss = -mixture_log_prob.mean()
            
            # Quality regularization: encourage using higher quality flows
            quality_reg = -self.quality_weight * (normalized_weights * per_sample_quality).sum(dim=1).mean()
            
            loss = base_loss + quality_reg
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f} (Base: {base_loss.item():.4f}, Quality: {quality_reg.item():.4f})")
        
        print(f"  Final loss: {losses[-1]:.4f}")
        self.meta_trained = True
        return losses
    
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
        """Forward pass with quality-aware adaptive meta-learner weights."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get flow predictions
        flow_predictions = self.get_flow_predictions(x)
        
        if self.meta_trained:
            # Use meta-learner for adaptive weights with quality awareness
            per_sample_quality = self.compute_quality_scores(x, per_sample=True)
            avg_quality_per_sample = per_sample_quality.mean(dim=1, keepdim=True)
            
            meta_input = torch.cat([x, flow_predictions, avg_quality_per_sample], dim=1)
            weights = self.meta_learner(meta_input)
            
            # Apply quality-aware reweighting
            quality_weighted = weights * per_sample_quality
            final_weights = quality_weighted / (quality_weighted.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # Fall back to uniform weights
            batch_size = x.size(0)
            final_weights = torch.ones(batch_size, len(self.flows), device=x.device) / len(self.flows)
        
        # Compute mixture log probability
        weighted_log_probs = flow_predictions + torch.log(final_weights + 1e-8)
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return {
            'log_prob': flow_predictions,
            'weights': final_weights,
            'mixture_log_prob': mixture_log_prob,
            'quality_scores': per_sample_quality if self.meta_trained else None,
        }
    
    def sample(self, n_samples):
        """Sample from the mixture using quality-aware adaptive weights."""
        device = next(self.parameters()).device
        
        if not self.meta_trained:
            # Fall back to uniform sampling if meta-learner not trained
            print("⚠️  Meta-learner not trained, using uniform sampling")
            return self._uniform_sample(n_samples)
        
        # Adaptive sampling using quality-aware meta-learner weights
        print("🎯 Using quality-aware adaptive sampling")
        
        # Generate a grid of sample points to estimate adaptive weights
        n_grid = min(500, n_samples)
        
        # Sample uniformly in a reasonable range to estimate weight distribution
        grid_range = 3.0
        grid_x = torch.linspace(-grid_range, grid_range, int(np.sqrt(n_grid)), device=device)
        grid_y = torch.linspace(-grid_range, grid_range, int(np.sqrt(n_grid)), device=device)
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_points = torch.stack([grid_xx.flatten(), grid_yy.flatten()], dim=1)[:n_grid]
        
        # Get quality-aware adaptive weights for grid points
        self.eval()
        with torch.no_grad():
            output = self.forward(grid_points)
            grid_weights = output['weights']  # [n_grid, n_flows]
        
        # Compute quality-weighted average weights
        quality_scores = self.compute_quality_scores(grid_points, per_sample=True)
        quality_weighted = grid_weights * quality_scores
        avg_weights = quality_weighted.mean(dim=0)  # [n_flows]
        avg_weights = avg_weights / (avg_weights.sum() + 1e-8)  # Normalize
        
        # Sample from each flow according to quality-aware adaptive weights
        all_samples = []
        for i, flow in enumerate(self.flows):
            # Number of samples for this flow based on quality-aware weight
            n_flow_samples = int(np.round(avg_weights[i].item() * n_samples))
            
            if n_flow_samples > 0:
                flow.eval()
                with torch.no_grad():
                    flow_samples = flow.sample(n_flow_samples)
                    all_samples.append(flow_samples)
        
        # Handle rounding errors - ensure we have exactly n_samples
        total_generated = sum(samples.size(0) for samples in all_samples)
        if total_generated < n_samples:
            # Add remaining samples from the flow with highest weight
            best_flow_idx = torch.argmax(avg_weights).item()
            remaining = n_samples - total_generated
            with torch.no_grad():
                extra_samples = self.flows[best_flow_idx].sample(remaining)
                all_samples.append(extra_samples)
        elif total_generated > n_samples:
            # Remove excess samples randomly
            excess = total_generated - n_samples
            if len(all_samples) > 0 and all_samples[-1].size(0) > excess:
                all_samples[-1] = all_samples[-1][:-excess]
        
        # Combine all samples
        if len(all_samples) == 0:
            # Fallback if no samples generated
            return self._uniform_sample(n_samples)
        
        final_samples = torch.cat(all_samples, dim=0)
        
        # Shuffle to avoid flow ordering bias
        perm = torch.randperm(final_samples.size(0), device=device)
        final_samples = final_samples[perm]
        
        print(f"📊 Quality-aware sampling weights: {[f'{w:.3f}' for w in avg_weights.cpu().numpy()]}")
        print(f"🏆 Flow quality scores: {[f'{score:.3f}' for score in self.flow_quality_scores.cpu().numpy()]}")
        
        return final_samples[:n_samples]  # Ensure exact count
    
    def _uniform_sample(self, n_samples):
        """Fallback uniform sampling method."""
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


def train_sequential_amf_vi(dataset_name='multimodal', show_plots=True, save_plots=False, temperature=1.0, quality_weight=0.1):
    """Train sequential AMF-VI with quality-aware meta-learner weights."""
    
    print(f"🚀 Quality-Aware Sequential AMF-VI on {dataset_name}")
    print(f"🌡️  Temperature: {temperature}, Quality Weight: {quality_weight}")
    print("=" * 60)
    
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Create sequential model with quality awareness
    model = SequentialAMFVI(
        dim=2, 
        flow_types=['realnvp', 'maf', 'iaf'],
        temperature=temperature,
        quality_weight=quality_weight
    )
    model = model.to(device)
    
    # Stage 1: Train flows independently
    flow_losses = model.train_flows_independently(data, epochs=200, lr=1e-3)
    
    # Stage 2: Train quality-aware meta-learner
    meta_losses = model.train_meta_learner(data, epochs=200, lr=1e-3)
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples using quality-aware adaptive sampling
        model_samples = model.sample(1000)
        
        # Individual flow samples for comparison
        flow_samples = {}
        flow_names = ['realnvp', 'maf', 'iaf']
        for i, name in enumerate(flow_names):
            flow_samples[name] = model.flows[i].sample(1000)
        
        # Get sample weights and quality analysis
        sample_output = model.forward(data[:100])  # First 100 samples
        avg_weights = sample_output['weights'].mean(dim=0).cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot target data
        data_np = data.cpu().numpy()
        axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
        axes[0, 0].set_title('Target Data')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot quality-aware model samples
        model_np = model_samples.cpu().numpy()
        axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
        axes[0, 1].set_title('Quality-Aware AMF-VI Samples')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot individual flows
        colors = ['green', 'orange', 'purple']
        for i, (name, samples) in enumerate(flow_samples.items()):
            if i < 3:
                row, col = (0, 2) if i == 0 else (1, i-1)
                samples_np = samples.cpu().numpy()
                axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                     alpha=0.6, c=colors[i], s=20)
                quality_score = model.flow_quality_scores[i].item()
                axes[row, col].set_title(f'{name.upper()} Flow (Q: {quality_score:.3f})')
                axes[row, col].grid(True, alpha=0.3)
        
        # Plot meta-learner training loss
        if meta_losses:
            axes[1, 2].plot(meta_losses, label='Quality-Aware Meta-Learner', color='red', linewidth=2)
            axes[1, 2].set_title('Meta-Learner Training Loss')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.suptitle(f'Quality-Aware Sequential AMF-VI - {dataset_name.title()}', fontsize=16)
        
        # Save plot if requested
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'quality_aware_amf_vi_results_{dataset_name}.png')
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
        print(f"Quality-aware model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        print(f"Target data std: {data.std(dim=0).cpu().numpy()}")
        print(f"Quality-aware model std: {model_samples.std(dim=0).cpu().numpy()}")
        
        # Quality-aware meta-learner analysis
        print(f"\n🧠 Quality-Aware Meta-Learner Analysis:")
        for i, name in enumerate(flow_names):
            weight = avg_weights[i]
            quality = model.flow_quality_scores[i].item()
            print(f"{name.upper()}: Weight={weight:.3f}, Quality={quality:.3f}, QxW={weight*quality:.3f}")
        
        # Check flow specialization
        print("\n🔍 Flow Specialization Analysis:")
        for i, (name, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            quality = model.flow_quality_scores[i].item()
            print(f"{name.upper()}: Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}], Quality={quality:.3f}")
        
        # Model complexity analysis
        print("\n🏗️ Model Architecture:")
        total_params = 0
        for i, flow in enumerate(model.flows):
            n_params = sum(p.numel() for p in flow.parameters())
            total_params += n_params
            print(f"{flow_names[i].upper()}: {n_params:,} parameters")
        
        # Meta-learner parameters
        meta_params = sum(p.numel() for p in model.meta_learner.parameters())
        total_params += meta_params
        print(f"QUALITY-AWARE META-LEARNER: {meta_params:,} parameters")
        print(f"Total parameters: {total_params:,}")
    
    # Save trained model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_quality_aware_{dataset_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'flow_losses': flow_losses,
            'meta_losses': meta_losses,
            'dataset': dataset_name,
            'temperature': temperature,
            'quality_weight': quality_weight
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, meta_losses


if __name__ == "__main__":
    # Run the quality-aware sequential AMF-VI experiment on multiple datasets
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    # Test different temperature and quality weight settings
    temperature_settings = [0.5, 1.0, 2.0]
    quality_weight_settings = [0.0, 0.1, 0.3]
    
    for dataset_name in datasets:
        for temp in temperature_settings:
            for quality_w in quality_weight_settings:
                print(f"\n{'='*80}")
                print(f"Training on dataset: {dataset_name.upper()}")
                print(f"Temperature: {temp}, Quality Weight: {quality_w}")
                print(f"{'='*80}")
                
                model, flow_losses, meta_losses = train_sequential_amf_vi(
                    dataset_name, 
                    show_plots=False, 
                    save_plots=True,
                    temperature=temp,
                    quality_weight=quality_w
                )