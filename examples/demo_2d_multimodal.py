import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from amf_vi.model import SimpleAMFVI
from amf_vi.loss import SimpleIWAELoss
from amf_vi.utils import (
    create_multimodal_data, 
    multimodal_log_prob, 
    plot_comparison,
    plot_comprehensive_results,
    plot_training_progress,
    plot_density_heatmap,
    plot_flow_weights_distribution,
    save_all_plots
)
import pickle
import os
from data.data_generator import generate_data

def check_for_nan(tensor, name="tensor"):
    """Diagnostic function to check for NaN values."""
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"  Min: {tensor[~torch.isnan(tensor)].min().item() if not torch.isnan(tensor).all() else 'All NaN'}")
        print(f"  Max: {tensor[~torch.isnan(tensor)].max().item() if not torch.isnan(tensor).all() else 'All NaN'}")
        return True
    return False

def train_amf_vi(show_plots=True, save_plots=False):
    """Train AMF-VI on 2D multimodal data with comprehensive visualization."""
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🚀 Starting AMF-VI training on 2D multimodal data...")
    
    # Create data
    data = generate_data('banana', n_samples=2000)
    #data = generate_data('x_shape', n_samples=1500, noise=0.15)
    
    # Create model with different flow types
    model = SimpleAMFVI(
        dim=2, 
        flow_types=['realnvp', 'planar', 'radial'],
        n_components=3
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)
    
    print(f"🏗️ Model created with {len(model.flows)} flows: {['realnvp', 'planar', 'radial']}")
    
    # Setup training with lower learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  #lr = 5e-4 # Lower learning rate
    
    # Add scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)
    
    loss_fn = SimpleIWAELoss(n_importance_samples=10) #5
    
    # Training loop with loss tracking
    n_epochs = 500
    batch_size = 64
    losses = []
    epoch_losses = []
    
    print(f"🎯 Training for {n_epochs} epochs with batch size {batch_size}")
    
    for epoch in range(n_epochs):
        # Mini-batch training
        perm = torch.randperm(len(data), device=device)
        total_loss = 0
        nan_detected = False
        batch_count = 0
        
        for i in range(0, len(data), batch_size):
            indices = perm[i:i+batch_size]
            batch = data[indices].to(device) 
            
            # Check input for NaN
            if check_for_nan(batch, "input batch"):
                print(f"❌ NaN in input at epoch {epoch}, batch {i//batch_size}")
                break
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Check model output for NaN
            for key, value in output.items():
                if check_for_nan(value, f"model output '{key}'"):
                    nan_detected = True
                    break
            
            if nan_detected:
                print(f"❌ NaN detected in model output at epoch {epoch}, batch {i//batch_size}")
                break
            
            # Compute loss
            main_loss = loss_fn(output, multimodal_log_prob, batch)
            reg_loss = model.regularization_loss()
            
            # Check losses for NaN
            if check_for_nan(main_loss, "main_loss") or check_for_nan(reg_loss, "reg_loss"):
                print(f"❌ NaN in loss at epoch {epoch}, batch {i//batch_size}")
                break
            
            loss = main_loss + reg_loss
            
            # Additional stability check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Invalid loss at epoch {epoch}: {loss.item()}")
                break
            
            # Backward pass with gradient clipping
            loss.backward()
            
            # FIX 1: More aggressive gradient clipping and NaN check
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Check for extremely small gradients that can become NaN
            if total_norm < 1e-10:
                print(f"⚠️ Extremely small gradients detected at epoch {epoch}, skipping step")
                continue
            
            # Check gradients for NaN after clipping
            grad_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"❌ NaN in gradient {name} at epoch {epoch}, batch {i//batch_size}")
                        grad_nan = True
                        break
            
            if grad_nan:
                print(f"❌ NaN in gradients at epoch {epoch}, batch {i//batch_size}")
                break
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if nan_detected or grad_nan:
            print(f"⚠️ Training stopped due to NaN at epoch {epoch}")
            break
        
        # Record epoch loss
        avg_epoch_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        epoch_losses.append(avg_epoch_loss)
        losses.extend([avg_epoch_loss])  # For compatibility with plotting functions
        
        # scheduler to update lr
        scheduler.step()
        
        if epoch % 50 == 0:
            print(f"📈 Epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
            
            # Additional diagnostics every 50 epochs
            with torch.no_grad():
                sample_output = model(data[:100])  # Small sample
                mixture_log_prob = model.log_prob_mixture(data[:100])
                if not check_for_nan(mixture_log_prob, "mixture_log_prob"):
                    print(f"   Log prob stats - Min: {mixture_log_prob.min():.3f}, Max: {mixture_log_prob.max():.3f}")
    
    print("✅ Training completed!")
    
    # Generate samples and create comprehensive visualizations
    print("🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Test model before visualization
        try:
            test_samples = model.sample(10)
            if check_for_nan(test_samples, "test samples"):
                print("⚠️ Model generates NaN samples, visualization may be limited")
                return model, losses
        except Exception as e:
            print(f"⚠️ Error testing model: {e}")
            return model, losses
        
        # Generate samples for visualization
        model_samples = model.sample(1000)
        
        # Check generated samples
        if check_for_nan(model_samples, "generated samples"):
            print("⚠️ Generated samples contain NaN")
            return model, losses
        
        # Individual flow samples
        flow_samples = {}
        flow_names = ['realnvp', 'planar', 'radial']
        for i, name in enumerate(flow_names):
            if i < len(model.flows):
                try:
                    samples = model.flows[i].sample(1000)
                    if not check_for_nan(samples, f"{name} flow samples"):
                        flow_samples[name] = samples
                except Exception as e:
                    print(f"⚠️ Error sampling from {name} flow: {e}")
        
        # FIX 2: Add .cpu() to convert CUDA tensors to CPU for numpy
        print(f"\n📊 Final Results:")
        print(f"   Target data mean: {data.mean(dim=0).cpu().numpy()}")
        print(f"   Model samples mean: {model_samples.mean(dim=0).cpu().numpy()}")
        print(f"   Target data std: {data.std(dim=0).cpu().numpy()}")
        print(f"   Model samples std: {model_samples.std(dim=0).cpu().numpy()}")
        
        if show_plots:
            print("📊 Displaying comprehensive results...")
            
            # 1. Main comprehensive plot
            try:
                fig_comprehensive = plot_comprehensive_results(data, model, epoch_losses)
                plt.show()
            except Exception as e:
                print(f"⚠️ Error creating comprehensive plot: {e}")
                
                # Fallback to basic comparison
                if len(flow_samples) > 0:
                    fig_basic = plot_comparison(data, model_samples, flow_samples)
                    plt.show()
            
            # 2. Training progress
            if len(epoch_losses) > 1:
                fig_loss = plot_training_progress(epoch_losses, "AMF-VI Training Loss")
                plt.show()
            
            # 3. Density heatmap
            try:
                fig_density = plot_density_heatmap(model)
                plt.show()
            except Exception as e:
                print(f"⚠️ Could not generate density heatmap: {e}")
            
            # 4. Flow weights distribution
            try:
                fig_weights = plot_flow_weights_distribution(model, data[:500])
                plt.show()
            except Exception as e:
                print(f"⚠️ Could not generate flow weights plot: {e}")
        
        if save_plots:
            print("💾 Saving all plots...")
            try:
                save_all_plots(data, model, epoch_losses)
                print("✅ All plots saved successfully!")
            except Exception as e:
                print(f"⚠️ Error saving plots: {e}")

    # Save trained model - FIX: Use relative path instead of absolute
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'trained_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")

    return model, epoch_losses

if __name__ == "__main__":
    # You can control visualization options here
    model, losses = train_amf_vi(
        show_plots=True,    # Set to False to skip showing plots
        save_plots=True     # Set to True to save plots to ./results/
    )