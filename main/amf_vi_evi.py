import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.distance import cdist

class ParticleEVI:
    """Particle-based Energetic Variational Inference for sample refinement."""
    
    def __init__(self, kernel_bandwidth=0.1, step_size=0.01, max_iterations=50):
        self.h = kernel_bandwidth
        self.tau = step_size
        self.max_iter = max_iterations
    
    def gaussian_kernel(self, x1, x2):
        """Gaussian kernel function."""
        dist_sq = torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0))**2, dim=2)
        return torch.exp(-dist_sq / (2 * self.h**2)) / ((2 * np.pi * self.h**2)**(x1.shape[1]/2))
    
    def compute_discrete_kl_energy(self, particles, target_log_prob_fn):
        """
        Compute discrete KL-divergence energy F_h as in Equation (3.10).
        
        Args:
            particles: torch.Tensor [N, dim] - current particle positions
            target_log_prob_fn: function that returns log probability of target
        """
        N = particles.shape[0]
        
        # Compute kernel matrix K_h(x_i, x_j)
        K = self.gaussian_kernel(particles, particles)
        
        # Compute log of kernel density estimate
        kernel_density = K.mean(dim=1)  # Average over j
        log_kernel_density = torch.log(kernel_density + 1e-8)
        
        # Compute target potential V(x_i) = -log ρ*(x_i)
        target_log_probs = target_log_prob_fn(particles)
        V = -target_log_probs
        
        # Discrete energy F_h (Equation 3.10)
        energy = (log_kernel_density + V).mean()
        
        return energy
    
    def compute_energy_gradient(self, particles, target_log_prob_fn):
        """
        Compute gradient of discrete energy F_h with respect to particles.
        This implements the gradient from Equation (3.13).
        """
        N = particles.shape[0]
        dim = particles.shape[1]
        
        # Enable gradient computation
        particles_grad = particles.clone().requires_grad_(True)
        
        # Compute kernel matrix and its gradients
        K = self.gaussian_kernel(particles_grad, particles_grad)
        
        # Compute target potential gradients
        target_log_probs = target_log_prob_fn(particles_grad)
        V = -target_log_probs
        
        # Compute total energy
        kernel_density = K.mean(dim=1)
        log_kernel_density = torch.log(kernel_density + 1e-8)
        energy = (log_kernel_density + V).mean()
        
        # Compute gradients
        energy.backward()
        gradients = particles_grad.grad.clone()
        
        return gradients
    
    def implicit_euler_step(self, particles, target_log_prob_fn):
        """
        Perform one implicit Euler step as in Algorithm 1.
        Solves the optimization problem J_n in Equation (3.18).
        """
        N, dim = particles.shape
        
        def objective_fn(new_particles):
            """Objective function J_n from Equation (3.17)"""
            # Regularization term: (1/2τ) * ||x - x^n||^2 / N
            reg_term = torch.sum((new_particles - particles)**2) / (2 * self.tau * N)
            
            # Energy term: F_h({x_i})
            energy_term = self.compute_discrete_kl_energy(new_particles, target_log_prob_fn)
            
            return reg_term + energy_term
        
        # Initialize optimization
        new_particles = particles.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([new_particles], lr=0.1, max_iter=20)
        
        def closure():
            optimizer.zero_grad()
            loss = objective_fn(new_particles)
            loss.backward()
            return loss
        
        # Optimize
        optimizer.step(closure)
        
        return new_particles.detach()
    
    def refine_samples(self, initial_samples, target_log_prob_fn, verbose=False):
        """
        Refine samples using particle-based EVI.
        
        Args:
            initial_samples: torch.Tensor [N, dim] - initial samples to refine
            target_log_prob_fn: function that computes log probability of target
            verbose: bool - whether to print progress
            
        Returns:
            refined_samples: torch.Tensor [N, dim] - refined samples
            energy_history: list - energy values during optimization
        """
        particles = initial_samples.clone()
        energy_history = []
        
        for iteration in range(self.max_iter):
            # Compute current energy
            current_energy = self.compute_discrete_kl_energy(particles, target_log_prob_fn)
            energy_history.append(current_energy.item())
            
            if verbose and iteration % 10 == 0:
                print(f"EVI Iteration {iteration}: Energy = {current_energy:.6f}")
            
            # Perform implicit Euler step
            particles_new = self.implicit_euler_step(particles, target_log_prob_fn)
            
            # Check convergence
            particle_change = torch.norm(particles_new - particles)
            if particle_change < 1e-6:
                if verbose:
                    print(f"EVI converged at iteration {iteration}")
                break
                
            particles = particles_new
        
        return particles, energy_history

class EnhancedSequentialAMFVI(nn.Module):
    """Enhanced Sequential AMF-VI with EVI post-processing."""
    
    def __init__(self, dim=2, flow_types=None, use_quality_weights=True, 
                 initial_temperature=2.0, final_temperature=0.8, use_evi_refinement=True):
        super().__init__()
        self.dim = dim
        self.use_quality_weights = use_quality_weights
        self.use_evi_refinement = use_evi_refinement
        
        # Temperature control parameters
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.current_temperature = initial_temperature
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf', 'iaf']
        
        # Create flows (assuming these classes exist)
        self.flows = nn.ModuleList()
        # Note: You'll need to import the actual flow classes
        # for flow_type in flow_types:
        #     if flow_type == 'realnvp':
        #         self.flows.append(RealNVPFlow(dim, n_layers=4))
        #     elif flow_type == 'maf':
        #         self.flows.append(MAFFlow(dim, n_layers=4))
        #     elif flow_type == 'iaf':
        #         self.flows.append(IAFFlow(dim, n_layers=4))
        
        # Initialize EVI refiner
        if self.use_evi_refinement:
            self.evi_refiner = ParticleEVI(kernel_bandwidth=0.1, step_size=0.01, max_iterations=30)
        
        self.flows_trained = False
        self.training_data = None
    
    def create_target_log_prob_fn(self, test_data):
        """Create a target log probability function for EVI refinement."""
        def target_log_prob_fn(particles):
            """
            Estimate target log probability using kernel density estimation
            on test data or using the mixture model.
            """
            # Option 1: Use mixture model predictions
            if hasattr(self, 'flows') and self.flows_trained:
                flow_predictions = self.get_flow_predictions(particles)
                if self.use_quality_weights:
                    flow_weights = self.compute_flow_quality_weights()
                    batch_size = particles.size(0)
                    weights = flow_weights.unsqueeze(0).expand(batch_size, -1)
                else:
                    batch_size = particles.size(0)
                    weights = torch.ones(batch_size, len(self.flows), device=particles.device) / len(self.flows)
                
                weighted_log_probs = flow_predictions + torch.log(weights + 1e-8)
                mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
                return mixture_log_prob
            
            # Option 2: Use kernel density estimation on test data
            else:
                # Simple KDE-based target log probability
                h_kde = 0.1
                distances = torch.cdist(particles, test_data)
                kernel_values = torch.exp(-distances**2 / (2 * h_kde**2))
                density_estimates = kernel_values.mean(dim=1)
                return torch.log(density_estimates + 1e-8)
        
        return target_log_prob_fn
    
    def sample_with_evi_refinement(self, n_samples, test_data=None):
        """
        Sample from the mixture and optionally refine using EVI.
        
        Args:
            n_samples: int - number of samples to generate
            test_data: torch.Tensor - test data for creating target log prob function
            
        Returns:
            samples: torch.Tensor - final samples (refined or unrefined)
            refinement_info: dict - information about EVI refinement if used
        """
        device = next(self.parameters()).device
        
        # Generate initial samples using original mixture sampling
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
        
        initial_samples = torch.cat(all_samples, dim=0) if all_samples else torch.empty(0, self.dim, device=device)
        
        refinement_info = {'used_evi': False}
        
        # Apply EVI refinement if enabled
        if self.use_evi_refinement and test_data is not None:
            target_log_prob_fn = self.create_target_log_prob_fn(test_data)
            
            refined_samples, energy_history = self.evi_refiner.refine_samples(
                initial_samples, target_log_prob_fn, verbose=True
            )
            
            refinement_info = {
                'used_evi': True,
                'energy_history': energy_history,
                'initial_samples': initial_samples,
                'energy_reduction': energy_history[0] - energy_history[-1] if energy_history else 0
            }
            
            return refined_samples, refinement_info
        
        return initial_samples, refinement_info

def enhanced_evaluate_single_dataset(dataset_name, model_type="quality", initial_temp=2.0, final_temp=0.8, use_evi=True):
    """
    Enhanced evaluation with EVI refinement option.
    """
    print(f"\n{'='*50}")
    print(f"Enhanced Evaluation: {dataset_name.upper()} dataset ({model_type} model)")
    if use_evi:
        print("🔧 EVI refinement ENABLED")
    print(f"{'='*50}")
    
    # Create test data
    # test_data = generate_data(dataset_name, n_samples=2000)  # You'll need to implement this
    
    # Load your trained model (implementation depends on your setup)
    # model = load_trained_model(dataset_name, model_type, initial_temp, final_temp)
    
    # For demonstration, assuming you have the model and test_data
    # Replace with your actual implementation
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_data = test_data.to(device)
    
    with torch.no_grad():
        # Generate samples with optional EVI refinement
        if use_evi:
            generated_samples, refinement_info = model.sample_with_evi_refinement(2000, test_data)
            
            if refinement_info['used_evi']:
                print(f"✅ EVI refinement applied!")
                print(f"   Energy reduction: {refinement_info['energy_reduction']:.6f}")
                print(f"   Refinement iterations: {len(refinement_info['energy_history'])}")
        else:
            generated_samples = model.sample(2000)
            refinement_info = {'used_evi': False}
        
        # Compute metrics
        coverage = compute_coverage(test_data, generated_samples)
        quality = compute_quality(test_data, generated_samples)
        
        print(f"📊 Results:")
        print(f"   Coverage: {coverage:.4f}")
        print(f"   Quality: {quality:.4f}")
        
        if use_evi and refinement_info['used_evi']:
            # Compare with unrefined samples
            initial_samples = refinement_info['initial_samples']
            initial_coverage = compute_coverage(test_data, initial_samples)
            initial_quality = compute_quality(test_data, initial_samples)
            
            print(f"📈 EVI Improvement:")
            print(f"   Coverage improvement: {coverage - initial_coverage:.4f}")
            print(f"   Quality improvement: {quality - initial_quality:.4f}")
    
    return {
        'coverage': coverage,
        'quality': quality,
        'refinement_info': refinement_info,
        'samples': generated_samples
    }
    """
    
    # Placeholder return for demonstration
    return {
        'coverage': 0.0,
        'quality': 0.0,
        'refinement_info': {'used_evi': use_evi},
        'samples': None
    }

# Example usage:
if __name__ == "__main__":
    # Test the EVI refinement on a simple 2D example
    
    # Create some dummy data for demonstration
    torch.manual_seed(42)
    test_data = torch.randn(1000, 2)  # Target distribution samples
    initial_samples = torch.randn(500, 2) * 1.5  # Initial samples to refine
    
    # Create EVI refiner
    evi_refiner = ParticleEVI(kernel_bandwidth=0.1, step_size=0.01, max_iterations=20)
    
    # Define a simple target log probability function
    def target_log_prob_fn(particles):
        # Simple 2D Gaussian target
        return -0.5 * torch.sum(particles**2, dim=1)
    
    # Refine samples
    print("Testing EVI refinement on dummy data...")
    refined_samples, energy_history = evi_refiner.refine_samples(
        initial_samples, target_log_prob_fn, verbose=True
    )
    
    print(f"Initial energy: {energy_history[0]:.6f}")
    print(f"Final energy: {energy_history[-1]:.6f}")
    print(f"Energy reduction: {energy_history[0] - energy_history[-1]:.6f}")
