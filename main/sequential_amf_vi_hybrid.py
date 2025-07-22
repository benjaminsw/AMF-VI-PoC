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
    """Sequential training version of AMF-VI with hybrid training strategy."""
    
    def __init__(self, dim=2, flow_types=None):
        super().__init__()
        self.dim = dim
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf', 'iaf']
        
        # Create flows
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=8))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=3))
            elif flow_type == 'iaf':
                self.flows.append(IAFFlow(dim, n_layers=3))
            elif flow_type == 'spline':
                self.flows.append(SplineFlow(dim, n_layers=8))
        
        # Meta-learner for adaptive weights
        self.meta_learner = nn.Sequential(
            nn.Linear(dim + len(self.flows), 64),  # input + flow predictions
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.flows)),
            nn.Softmax(dim=1)
        )
        
        # Track training stages
        self.flows_trained = False
        self.meta_trained = False
        self.joint_trained = False  # NEW
    
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
    
    def train_meta_learner(self, data, epochs=300, lr=1e-3):
        """Stage 2: Train meta-learner to learn adaptive weights (flows frozen)."""
        print("🔄 Stage 2: Training meta-learner...")
        
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        # Freeze all flow parameters
        for flow in self.flows:
            for param in flow.parameters():
                param.requires_grad = False
        
        # Get flow predictions (fixed)
        flow_predictions = self.get_flow_predictions(data)
        
        # Create input for meta-learner: [x, flow_predictions]
        meta_input = torch.cat([data, flow_predictions], dim=1)
        
        # Train meta-learner only
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
        self.meta_trained = True
        return losses
    
    def fine_tune_joint(self, data, epochs=100, lr=1e-4):
        """Stage 3: Joint fine-tuning of all components (NEW)."""
        print("🔄 Stage 3: Joint fine-tuning...")
        
        if not self.flows_trained or not self.meta_trained:
            raise RuntimeError("Both flows and meta-learner must be trained first!")
        
        # Unfreeze all parameters for joint training
        for flow in self.flows:
            for param in flow.parameters():
                param.requires_grad = True
        
        for param in self.meta_learner.parameters():
            param.requires_grad = True
        
        # Create optimizer for all parameters
        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass through the entire system
            model_output = self.forward(data)
            
            # Joint loss (negative log-likelihood of mixture)
            loss = -model_output['mixture_log_prob'].mean()
            
            # Add small regularization to prevent catastrophic forgetting
            reg_loss = 0.01 * sum(torch.norm(p)**2 for p in self.parameters())
            total_loss = loss + reg_loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        print(f"  Final loss: {losses[-1]:.4f}")
        self.joint_trained = True
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
        """Forward pass with adaptive meta-learner weights."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get flow predictions (now potentially trainable in stage 3)
        if self.joint_trained:
            # In joint training mode, flows can be updated
            flow_log_probs = []
            for flow in self.flows:
                log_prob = flow.log_prob(x)
                flow_log_probs.append(log_prob.unsqueeze(1))
            flow_predictions = torch.cat(flow_log_probs, dim=1)
        else:
            # Use frozen flow predictions
            flow_predictions = self.get_flow_predictions(x)
        
        if self.meta_trained:
            # Use meta-learner for adaptive weights
            meta_input = torch.cat([x, flow_predictions], dim=1)
            weights = self.meta_learner(meta_input)
        else:
            # Fall back to uniform weights
            batch_size = x.size(0)
            weights = torch.ones(batch_size, len(self.flows), device=x.device) / len(self.flows)
        
        # Compute mixture log probability
        weighted_log_probs = flow_predictions + torch.log(weights + 1e-8)
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return {
            'log_prob': flow_predictions,
            'weights': weights,
            'mixture_log_prob': mixture_log_prob,
        }
    
    def sample(self, n_samples):
        """Sample from the mixture with uniform weights."""
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

def train_sequential_amf_vi(dataset_name='multimodal', show_plots=True, save_plots=False):
    """Train sequential AMF-VI with hybrid training strategy."""
    
    print(f"🚀 Sequential AMF-VI Hybrid Training on {dataset_name}")
    print("=" * 60)
    
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Create sequential model
    model = SequentialAMFVI(dim=2, flow_types=['realnvp', 'maf', 'iaf'])
    model = model.to(device)
    
    # Stage 1: Train flows independently
    flow_losses = model.train_flows_independently(data, epochs=200, lr=1e-3)
    
    # Stage 2: Train meta-learner
    meta_losses = model.train_meta_learner(data, epochs=200, lr=1e-3)
    
    # Stage 3: Joint fine-tuning (NEW)
    joint_losses = model.fine_tune_joint(data, epochs=100, lr=1e-4)
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000)
        
        # Individual flow samples
        flow_samples = {}
        flow_names = ['realnvp', 'maf', 'iaf']
        for i, name in enumerate(flow_names):
            flow_samples[name] = model.flows[i].sample(1000)
        
        # Get sample weights to analyze meta-learner behavior
        sample_output = model.forward(data[:100])  # First 100 samples
        avg_weights = sample_output['weights'].mean(dim=0).cpu().numpy()
        
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
        axes[0, 1].set_title('Hybrid AMF-VI Samples')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot individual flows
        colors = ['green', 'orange', 'purple']
        for i, (name, samples) in enumerate(flow_samples.items()):
            if i < 3:
                row, col = (0, 2) if i == 0 else (1, i-1)
                samples_np = samples.cpu().numpy()
                axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                     alpha=0.6, c=colors[i], s=20)
                axes[row, col].set_title(f'{name.upper()} Flow')
                axes[row, col].grid(True, alpha=0.3)
        
        # Plot all training losses (UPDATED)
        if meta_losses and joint_losses:
            axes[1, 2].plot(meta_losses, label='Meta-Learner', color='red', linewidth=2, alpha=0.8)
            # Offset joint losses to show after meta training
            joint_x = np.arange(len(meta_losses), len(meta_losses) + len(joint_losses))
            axes[1, 2].plot(joint_x, joint_losses, label='Joint Fine-tuning', color='purple', linewidth=2, alpha=0.8)
            axes[1, 2].axvline(x=len(meta_losses), color='gray', linestyle='--', alpha=0.5)
            axes[1, 2].text(len(meta_losses)//2, max(meta_losses)*0.9, 'Stage 2', ha='center', alpha=0.7)
            axes[1, 2].text(len(meta_losses) + len(joint_losses)//2, max(meta_losses)*0.9, 'Stage 3', ha='center', alpha=0.7)
        axes[1, 2].set_title('Hybrid Training Losses')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.suptitle(f'Sequential AMF-VI Hybrid Training - {dataset_name.title()}', fontsize=16)
        
        # Save plot if requested
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'sequential_amf_vi_hybrid_results_{dataset_name}.png')
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
        print(f"Hybrid model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        print(f"Target data std: {data.std(dim=0).cpu().numpy()}")
        print(f"Hybrid model std: {model_samples.std(dim=0).cpu().numpy()}")
        
        # Meta-learner weight analysis
        print(f"\n🧠 Meta-Learner Weight Analysis:")
        for i, name in enumerate(flow_names):
            print(f"{name.upper()} average weight: {avg_weights[i]:.3f}")
        
        # Check flow diversity
        print("\n🔍 Flow Specialization Analysis:")
        for i, (name, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            print(f"{name.upper()}: Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
        
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
        print(f"META-LEARNER: {meta_params:,} parameters")
        print(f"Total parameters: {total_params:,}")
        
        # Training stage summary (NEW)
        print(f"\n🏁 Training Summary:")
        print(f"Stage 1 (Flows): {len(flow_losses[0]) if flow_losses else 0} epochs")
        print(f"Stage 2 (Meta-learner): {len(meta_losses)} epochs") 
        print(f"Stage 3 (Joint): {len(joint_losses)} epochs")
        print(f"Total training epochs: {len(flow_losses[0]) + len(meta_losses) + len(joint_losses) if flow_losses else 0}")
    
    # Save trained model (UPDATED)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_hybrid_{dataset_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'flow_losses': flow_losses,
            'meta_losses': meta_losses,
            'joint_losses': joint_losses,  # NEW
            'dataset': dataset_name
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, meta_losses, joint_losses

if __name__ == "__main__":
    # Run the sequential AMF-VI experiment on multiple datasets
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different']
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        model, flow_losses, meta_losses, joint_losses = train_sequential_amf_vi(
            dataset_name, 
            show_plots=False, 
            save_plots=True
        )