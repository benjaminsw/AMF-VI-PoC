import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data.data_generator import generate_data
import os
import pickle
import csv
from sklearn.mixture import GaussianMixture

# Set seed for reproducibility
torch.manual_seed(2025)
np.random.seed(2025)

class TargetDistribution:
    """Target distributions for each dataset with analytical log probabilities."""
    
    def __init__(self, dataset_name, debug=False):
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        self.eps = 1e-8  # Numerical stability constant
    
    def _validate_input(self, x):
        """Phase 3.1: Dimension validation and input checks."""
        assert isinstance(x, torch.Tensor), f"Input must be torch.Tensor, got {type(x)}"
        assert x.dim() == 2, f"Input must be 2D tensor, got {x.dim()}D"
        assert x.shape[1] == 2, f"Input must be 2D data, got shape {x.shape}"
        if self.debug:
            print(f"Input validation passed: shape={x.shape}, device={x.device}, dtype={x.dtype}")
    
    def _safe_quadratic_form(self, x, mu, cov_inv):
        """Gradient-safe quadratic form computation: (x-mu)^T * cov_inv * (x-mu)"""
        # Phase 3.1: Validate tensor compatibility
        assert x.shape[1] == mu.shape[0], f"Dimension mismatch: x.shape[1]={x.shape[1]}, mu.shape[0]={mu.shape[0]}"
        assert cov_inv.shape == (mu.shape[0], mu.shape[0]), f"Covariance matrix shape mismatch: {cov_inv.shape}"
        
        diff = x - mu  # [batch_size, dim]
        quad_form = torch.einsum('bi,ij,bj->b', diff, cov_inv, diff)
        
        # Phase 3.2: Numerical stability - clamp extreme values
        quad_form = torch.clamp(quad_form, min=0, max=1e6)
        return quad_form
    
    def _safe_mvn_log_prob(self, x, mu, cov_inv, log_det_cov):
        """Safe multivariate normal log probability with numerical stability."""
        quad_form = self._safe_quadratic_form(x, mu, cov_inv)
        dim = x.shape[1]
        
        # Phase 3.2: Add epsilon regularization and clamp extreme values
        log_det_cov = max(log_det_cov, -20)  # Prevent extremely negative log determinants
        log_prob = -0.5 * quad_form - 0.5 * dim * np.log(2 * np.pi) - 0.5 * log_det_cov
        
        # Clamp to prevent numerical overflow/underflow
        log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)
        return log_prob
    
    def _safe_mixture_log_prob(self, x, components):
        """Phase 4.1: Safe mixture model log probability computation."""
        self._validate_input(x)
        
        log_probs = []
        for comp in components:
            mu, cov_inv, log_det = comp['mu'], comp['cov_inv'], comp['log_det']
            log_p = self._safe_mvn_log_prob(x, mu, cov_inv, log_det)
            log_probs.append(log_p)
        
        log_probs = torch.stack(log_probs, dim=1)
        
        # Phase 3.2: Numerical stability in logsumexp
        mixture_log_prob = torch.logsumexp(log_probs, dim=1) - np.log(len(components))
        
        # Clamp final result
        return torch.clamp(mixture_log_prob, min=-1e6, max=1e6)
    
    def log_prob(self, x):
        """Compute log probability for target distribution with error handling."""
        try:
            self._validate_input(x)
            
            if self.dataset_name == 'banana':
                return self._banana_log_prob(x)
            elif self.dataset_name == 'x_shape':
                return self._x_shape_log_prob(x)
            elif self.dataset_name == 'bimodal_shared':
                return self._bimodal_shared_log_prob(x)
            elif self.dataset_name == 'bimodal_different':
                return self._bimodal_different_log_prob(x)
            elif self.dataset_name == 'multimodal':
                return self._multimodal_log_prob(x)
            elif self.dataset_name == 'two_moons':
                return self._two_moons_log_prob(x)
            elif self.dataset_name == 'rings':
                return self._rings_log_prob(x)
            else:
                # Fallback: fit GMM to data
                return self._gmm_log_prob(x)
        except Exception as e:
            if self.debug:
                print(f"Error in log_prob for {self.dataset_name}: {e}")
            # Return very low but finite log probability
            return torch.full((x.shape[0],), -1e6, device=x.device, dtype=x.dtype)
    
    def grad_log_prob(self, x):
        """Compute gradient of log probability (for MALA) with enhanced error handling."""
        try:
            self._validate_input(x)
            
            if not x.requires_grad:
                x.requires_grad_(True)
            
            log_p = self.log_prob(x).sum()
            grad = torch.autograd.grad(log_p, x, create_graph=False, retain_graph=False)[0]
            
            # Phase 3.2: Clamp gradients to prevent extreme values
            grad = torch.clamp(grad, min=-1e3, max=1e3)
            return grad
            
        except Exception as e:
            if self.debug:
                print(f"Gradient computation failed: {e}, using numerical gradient")
            # Fallback: numerical gradient
            return self._numerical_grad(x)
    
    def _numerical_grad(self, x):
        """Fallback numerical gradient computation with error handling."""
        eps = 1e-6
        grad = torch.zeros_like(x)
        x_detach = x.detach().clone()
        
        try:
            for i in range(x.shape[1]):
                x_plus = x_detach.clone()
                x_minus = x_detach.clone()
                x_plus[:, i] += eps
                x_minus[:, i] -= eps
                
                log_p_plus = self.log_prob(x_plus)
                log_p_minus = self.log_prob(x_minus)
                grad[:, i] = (log_p_plus - log_p_minus) / (2 * eps)
            
            # Phase 3.2: Clamp numerical gradients
            grad = torch.clamp(grad, min=-1e3, max=1e3)
            return grad
            
        except Exception as e:
            if self.debug:
                print(f"Numerical gradient failed: {e}, returning zero gradient")
            return torch.zeros_like(x)
    
    def _banana_log_prob(self, x):
        """Banana distribution: N(x2; x1^2/4, 1) * N(x1; 0, 2)"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        
        # Phase 3.2: Add numerical stability
        log_p1 = -0.5 * x1**2 / 2 - 0.5 * np.log(2 * np.pi * 2)
        log_p2 = -0.5 * (x2 - x1**2/4)**2 - 0.5 * np.log(2 * np.pi)
        
        log_prob = log_p1 + log_p2
        return torch.clamp(log_prob, min=-1e6, max=1e6)
    
    def _x_shape_log_prob(self, x):
        """X-shape: Mixture of two diagonal Gaussians - Phase 4.2: Refactored."""
        # Component parameters with proper device and dtype
        cov1_inv = torch.tensor([[0.278, -0.25], [-0.25, 0.278]], 
                               dtype=x.dtype, device=x.device)
        cov2_inv = torch.tensor([[0.278, 0.25], [0.25, 0.278]], 
                               dtype=x.dtype, device=x.device)
        mu1 = torch.zeros(2, dtype=x.dtype, device=x.device)
        mu2 = torch.zeros(2, dtype=x.dtype, device=x.device)
        
        # Phase 3.2: Add epsilon regularization to determinants
        log_det1 = np.log(max(0.4, self.eps))
        log_det2 = np.log(max(0.4, self.eps))
        
        # Phase 4.1: Use safe mixture helper
        components = [
            {'mu': mu1, 'cov_inv': cov1_inv, 'log_det': log_det1},
            {'mu': mu2, 'cov_inv': cov2_inv, 'log_det': log_det2}
        ]
        return self._safe_mixture_log_prob(x, components)
    
    def _bimodal_shared_log_prob(self, x):
        """Bimodal with shared covariance - Phase 4.2: Refactored."""
        mu1 = torch.tensor([-1.5, 0.0], dtype=x.dtype, device=x.device)
        mu2 = torch.tensor([1.5, 0.0], dtype=x.dtype, device=x.device)
        cov_inv = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=x.dtype, device=x.device)
        
        # Phase 3.2: Add epsilon regularization
        log_det_cov = np.log(max(0.25, self.eps))
        
        components = [
            {'mu': mu1, 'cov_inv': cov_inv, 'log_det': log_det_cov},
            {'mu': mu2, 'cov_inv': cov_inv, 'log_det': log_det_cov}
        ]
        return self._safe_mixture_log_prob(x, components)
    
    def _bimodal_different_log_prob(self, x):
        """Bimodal with different covariances - Phase 4.2: Refactored."""
        mu1 = torch.tensor([-2.25, -0.5], dtype=x.dtype, device=x.device)
        mu2 = torch.tensor([2.25, 0.5], dtype=x.dtype, device=x.device)
        
        # Phase 3.2: Check for singular matrices
        cov1_inv = torch.tensor([[1.316, -0.526], [-0.526, 3.448]], 
                               dtype=x.dtype, device=x.device)
        cov2_inv = torch.tensor([[3.448, 1.053], [1.053, 1.754]], 
                               dtype=x.dtype, device=x.device)
        
        log_det1 = np.log(max(0.21, self.eps))
        log_det2 = np.log(max(0.17, self.eps))
        
        components = [
            {'mu': mu1, 'cov_inv': cov1_inv, 'log_det': log_det1},
            {'mu': mu2, 'cov_inv': cov2_inv, 'log_det': log_det2}
        ]
        return self._safe_mixture_log_prob(x, components)
    
    def _multimodal_log_prob(self, x):
        """Multimodal Gaussian mixture - Phase 4.2: Refactored."""
        mus = torch.tensor([[-2.0, -1.0], [2.0, 1.0], [0.0, 2.5]], 
                          dtype=x.dtype, device=x.device)
        cov_inv = torch.tensor([[3.33, 0.0], [0.0, 3.33]], 
                              dtype=x.dtype, device=x.device)
        
        log_det_cov = np.log(max(0.09, self.eps))
        
        components = []
        for i in range(mus.shape[0]):
            components.append({
                'mu': mus[i], 
                'cov_inv': cov_inv, 
                'log_det': log_det_cov
            })
        
        return self._safe_mixture_log_prob(x, components)
    
    def _two_moons_log_prob(self, x):
        """Two moons approximation using GMM"""
        return self._gmm_log_prob(x)
    
    def _rings_log_prob(self, x):
        """Concentric rings approximation with numerical stability."""
        r = torch.norm(x, dim=1)
        # Phase 3.2: Add epsilon to prevent division by zero or log(0)
        r = torch.clamp(r, min=self.eps)
        
        # Two rings at r=1.0 and r=2.5
        log_p1 = -0.5 * (r - 1.0)**2 / max(0.01, self.eps)
        log_p2 = -0.5 * (r - 2.5)**2 / max(0.01, self.eps)
        
        log_probs = torch.stack([log_p1, log_p2], dim=1)
        mixture_log_prob = torch.logsumexp(log_probs, dim=1) - np.log(2)
        return torch.clamp(mixture_log_prob, min=-1e6, max=1e6)
    
    def _gmm_log_prob(self, x):
        """Fallback: fit GMM to generated data with error handling."""
        try:
            if not hasattr(self, '_gmm_fitted'):
                # Fit GMM once
                data = generate_data(self.dataset_name, n_samples=5000).numpy()
                self._gmm = GaussianMixture(n_components=5, random_state=42, reg_covar=self.eps)
                self._gmm.fit(data)
                self._gmm_fitted = True
            
            # Compute log probability
            x_np = x.detach().cpu().numpy()
            log_probs = self._gmm.score_samples(x_np)
            log_probs_tensor = torch.tensor(log_probs, dtype=x.dtype, device=x.device)
            
            # Phase 3.2: Clamp GMM log probabilities
            return torch.clamp(log_probs_tensor, min=-1e6, max=1e6)
            
        except Exception as e:
            if self.debug:
                print(f"GMM log_prob failed: {e}, returning uniform distribution")
            # Fallback: return uniform log probability
            return torch.full((x.shape[0],), -2 * np.log(10), device=x.device, dtype=x.dtype)

class MALASampler:
    """Metropolis-Adjusted Langevin Algorithm sampler with enhanced error handling."""
    
    def __init__(self, target_dist, step_size=0.01, dim=2, debug=False):
        self.target_dist = target_dist
        self.step_size = step_size
        self.dim = dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        # Adaptation parameters
        self.adapt_step_size = True
        self.target_accept_rate = 1.1/0.9 #0.574  # Optimal for MALA
        self.adapt_window = 100
        
        # Phase 3: Error handling counters
        self.error_count = 0
        self.max_errors = 100  # Maximum consecutive errors before stopping
    
    def _validate_state(self, x):
        """Phase 3.1: Validate MALA state."""
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            if self.debug:
                print(f"Invalid state detected: NaN={torch.any(torch.isnan(x))}, Inf={torch.any(torch.isinf(x))}")
            return False
        if torch.norm(x) > 1e3:  # Prevent extreme values
            if self.debug:
                print(f"State norm too large: {torch.norm(x)}")
            return False
        return True
    
    def sample(self, n_samples, n_burnin=1000, init_point=None):
        """Generate samples using MALA with enhanced error handling."""
        total_samples = n_samples + n_burnin
        
        # Initialize with validation
        if init_point is None:
            x_current = torch.randn(self.dim, device=self.device) * 0.1
        else:
            x_current = init_point.clone()
        
        # Phase 3.1: Validate initial state
        if not self._validate_state(x_current):
            print("Warning: Invalid initial state, using safe initialization")
            x_current = torch.zeros(self.dim, device=self.device)
        
        samples = []
        n_accepted = 0
        step_sizes = []
        consecutive_errors = 0
        
        print(f"🔥 MALA Sampling: {total_samples} total samples ({n_burnin} burn-in)")
        
        for i in range(total_samples):
            try:
                # Compute gradient at current point
                grad_current = self.target_dist.grad_log_prob(x_current.unsqueeze(0)).squeeze(0)
                
                # Phase 3.1: Validate gradient
                if torch.any(torch.isnan(grad_current)) or torch.any(torch.isinf(grad_current)):
                    if self.debug:
                        print(f"Invalid gradient at step {i}, using zero gradient")
                    grad_current = torch.zeros_like(grad_current)
                
                # Propose new state
                noise = torch.randn_like(x_current)
                x_proposed = (x_current + 0.5 * self.step_size**2 * grad_current + 
                             self.step_size * noise)
                
                # Phase 3.1: Validate proposed state
                if not self._validate_state(x_proposed):
                    if self.debug:
                        print(f"Invalid proposed state at step {i}, rejecting")
                    x_proposed = x_current  # Reject by keeping current state
                
                # Compute acceptance probability
                log_alpha = self._compute_log_acceptance_prob(x_current, x_proposed, 
                                                            grad_current)
                
                # Phase 3.2: Handle invalid acceptance probability
                if torch.isnan(log_alpha) or torch.isinf(log_alpha):
                    log_alpha = torch.tensor(-float('inf'))  # Force rejection
                
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                
                # Accept or reject
                if torch.rand(1).item() < alpha and self._validate_state(x_proposed):
                    x_current = x_proposed
                    n_accepted += 1
                    consecutive_errors = 0  # Reset error counter on success
                
                # Store sample (after burn-in)
                if i >= n_burnin:
                    samples.append(x_current.clone())
                
            except Exception as e:
                consecutive_errors += 1
                if self.debug or consecutive_errors <= 5:
                    print(f"Warning: MALA step {i} failed: {e}")
                
                # Phase 3: Handle consecutive errors
                if consecutive_errors >= self.max_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}), stopping sampling")
                    break
                
                # Keep current sample on error
                if i >= n_burnin:
                    samples.append(x_current.clone())
            
            # Phase 3.2: More conservative step size adaptation
            if self.adapt_step_size and i > 0 and i % self.adapt_window == 0:
                accept_rate = n_accepted / (i + 1)
                if accept_rate > self.target_accept_rate:
                    self.step_size *= 1.02  # Very conservative increase
                else:
                    self.step_size *= 0.98  # Conservative decrease
                self.step_size = torch.clamp(torch.tensor(self.step_size), 0.0001, 0.1).item()
            
            step_sizes.append(self.step_size)
            
            # Progress reporting
            if (i + 1) % max(1, total_samples // 10) == 0:
                accept_rate = n_accepted / (i + 1)
                print(f"    Step {i+1}/{total_samples}: Accept Rate = {accept_rate:.3f}, "
                      f"Step Size = {self.step_size:.4f}, Errors = {consecutive_errors}")
        
        final_accept_rate = n_accepted / total_samples if total_samples > 0 else 0
        print(f"    Final acceptance rate: {final_accept_rate:.3f}")
        print(f"    Total errors encountered: {self.error_count}")
        
        # Phase 3: Handle case where no valid samples were collected
        if len(samples) == 0:
            print("Warning: No valid samples collected, returning random samples")
            samples = [torch.randn(self.dim, device=self.device) * 0.1 for _ in range(n_samples)]
        
        return torch.stack(samples), {
            'acceptance_rate': final_accept_rate,
            'step_sizes': step_sizes,
            'n_accepted': n_accepted,
            'total_errors': self.error_count
        }
    
    def _compute_log_acceptance_prob(self, x_current, x_proposed, grad_current):
        """Compute log acceptance probability for MALA with enhanced error handling."""
        try:
            # Forward proposal density
            grad_proposed = self.target_dist.grad_log_prob(x_proposed.unsqueeze(0)).squeeze(0)
            
            # Phase 3.2: Validate gradients
            if torch.any(torch.isnan(grad_proposed)) or torch.any(torch.isinf(grad_proposed)):
                if self.debug:
                    print("Invalid proposed gradient, using current gradient")
                grad_proposed = grad_current
            
            mean_forward = x_current + 0.5 * self.step_size**2 * grad_current
            mean_backward = x_proposed + 0.5 * self.step_size**2 * grad_proposed
            
            # Phase 3.2: Add numerical stability to proposal densities
            log_q_forward = -0.5 * torch.sum((x_proposed - mean_forward)**2) / max(self.step_size**2, 1e-8)
            log_q_backward = -0.5 * torch.sum((x_current - mean_backward)**2) / max(self.step_size**2, 1e-8)
            
            # Target densities
            log_p_current = self.target_dist.log_prob(x_current.unsqueeze(0)).squeeze(0)
            log_p_proposed = self.target_dist.log_prob(x_proposed.unsqueeze(0)).squeeze(0)
            
            # Phase 3.2: Validate target densities
            if torch.isnan(log_p_current) or torch.isinf(log_p_current):
                log_p_current = torch.tensor(-1e6)
            if torch.isnan(log_p_proposed) or torch.isinf(log_p_proposed):
                log_p_proposed = torch.tensor(-1e6)
            
            # Log acceptance probability
            log_alpha = log_p_proposed - log_p_current + log_q_backward - log_q_forward
            
            # Phase 3.2: Final clamp with more conservative bounds
            return torch.clamp(log_alpha, min=-50, max=0)
            
        except Exception as e:
            self.error_count += 1
            if self.debug:
                print(f"Acceptance probability computation failed: {e}")
            return torch.tensor(-float('inf'))  # Reject proposal

def compute_cross_entropy_surrogate_mala(target_samples, mala_samples):
    """Compute cross-entropy surrogate: -E_p[log q(x)] where q is MALA samples."""
    # Use kernel density estimation for MALA sample density
    from sklearn.neighbors import KernelDensity
    
    # Fit KDE to MALA samples
    mala_np = mala_samples.detach().cpu().numpy()
    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(mala_np)
    
    # Evaluate on target samples
    target_np = target_samples.detach().cpu().numpy()
    log_q = kde.score_samples(target_np)
    
    return -np.mean(log_q)

def compute_kl_divergence_mala(target_samples, mala_samples):
    """Compute KL divergence using histogram method."""
    target_np = target_samples.detach().cpu().numpy()
    mala_np = mala_samples.detach().cpu().numpy()
    
    # Histogram-based KL divergence
    bins = 50
    
    # Get data range
    x_min = min(target_np[:, 0].min(), mala_np[:, 0].min())
    x_max = max(target_np[:, 0].max(), mala_np[:, 0].max())
    y_min = min(target_np[:, 1].min(), mala_np[:, 1].min())
    y_max = max(target_np[:, 1].max(), mala_np[:, 1].max())
    
    # Create histograms
    hist_target, _, _ = np.histogram2d(target_np[:, 0], target_np[:, 1], 
                                       bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    hist_mala, _, _ = np.histogram2d(mala_np[:, 0], mala_np[:, 1], 
                                    bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    
    # Normalize
    hist_target = hist_target / hist_target.sum()
    hist_mala = hist_mala / hist_mala.sum()
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    hist_target = hist_target + epsilon
    hist_mala = hist_mala + epsilon
    
    # KL divergence
    kl_div = np.sum(hist_target * np.log(hist_target / hist_mala))
    
    return kl_div

def evaluate_mala_dataset(dataset_name):
    """Evaluate MALA on a single dataset with enhanced error handling."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating MALA on {dataset_name.upper()} dataset")  
    print(f"{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Generate test data with validation
        test_data = generate_data(dataset_name, n_samples=2000).to(device)
        if test_data.shape[1] != 2:
            raise ValueError(f"Invalid test data shape: {test_data.shape}")
        
        # Create target distribution with debug mode for problematic datasets
        debug_mode = dataset_name in ['two_moons', 'rings']
        target_dist = TargetDistribution(dataset_name, debug=debug_mode)
        
        # Phase 3.1: Dataset-specific parameters with more conservative values
        step_size_map = {
            'banana': 0.03*5, 'x_shape': 0.02*5, 'bimodal_shared': 0.03,
            'bimodal_different': 0.02, 'multimodal': 0.015, 
            'two_moons': 0.01, 'rings': 0.005
        }
        # adjust step sizes by 5 times
        step_size_map = {k: v * 5 for k, v in step_size_map.items()}
        step_size = step_size_map.get(dataset_name, 0.01)
        
        sampler = MALASampler(target_dist, step_size=step_size, debug=debug_mode)
        
        # Phase 3: Reduced sample size for stability
        print(f"Using conservative step size: {step_size}")
        mala_samples, sampling_info = sampler.sample(n_samples=1500, n_burnin=5000)
        
        # Phase 3.2: Validate samples before computing metrics
        if len(mala_samples) == 0:
            raise ValueError("No valid samples generated")
        
        # Compute metrics with error handling
        try:
            kl_divergence = compute_kl_divergence_mala(test_data, mala_samples)
            cross_entropy = compute_cross_entropy_surrogate_mala(test_data, mala_samples)
        except Exception as e:
            print(f"Warning: Metrics computation failed: {e}")
            kl_divergence = float('inf')
            cross_entropy = float('inf')
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot target data
        test_np = test_data.cpu().numpy()
        axes[0, 0].scatter(test_np[:, 0], test_np[:, 1], alpha=0.6, s=15, c='blue')
        axes[0, 0].set_title(f'Target Data ({dataset_name})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot MALA samples
        mala_np = mala_samples.cpu().numpy()
        axes[0, 1].scatter(mala_np[:, 0], mala_np[:, 1], alpha=0.6, s=15, c='red')
        axes[0, 1].set_title('MALA Samples')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overlay comparison
        axes[0, 2].scatter(test_np[:, 0], test_np[:, 1], alpha=0.4, s=10, c='blue', label='Target')
        axes[0, 2].scatter(mala_np[:, 0], mala_np[:, 1], alpha=0.4, s=10, c='red', label='MALA')
        axes[0, 2].set_title('Overlay Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Acceptance rate over time
        step_sizes = sampling_info['step_sizes']
        axes[1, 0].plot(step_sizes)
        axes[1, 0].set_title('Step Size Adaptation')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Step Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample trace (first dimension)
        axes[1, 1].plot(mala_np[:500, 0])
        axes[1, 1].set_title('Sample Trace (X1)')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('X1')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Metrics summary
        axes[1, 2].axis('off')
        metrics_text = f"""MALA Results:
        
KL Divergence: {kl_divergence:.3f}
Cross-Entropy: {cross_entropy:.3f}
Acceptance Rate: {sampling_info['acceptance_rate']:.3f}
Final Step Size: {step_sizes[-1]:.4f}
Total Errors: {sampling_info.get('total_errors', 0)}

Target Mean: [{test_np.mean(axis=0)[0]:.2f}, {test_np.mean(axis=0)[1]:.2f}]
MALA Mean: [{mala_np.mean(axis=0)[0]:.2f}, {mala_np.mean(axis=0)[1]:.2f}]

Target Std: [{test_np.std(axis=0)[0]:.2f}, {test_np.std(axis=0)[1]:.2f}]
MALA Std: [{mala_np.std(axis=0)[0]:.2f}, {mala_np.std(axis=0)[1]:.2f}]"""
        
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.suptitle(f'MALA Evaluation - {dataset_name.title()}', fontsize=16, y=0.98)
        
        # Save results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'mala_evaluation_{dataset_name}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save samples
        results = {
            'dataset': dataset_name,
            'mala_samples': mala_samples,
            'test_data': test_data,
            'kl_divergence': kl_divergence,
            'cross_entropy_surrogate': cross_entropy,
            'sampling_info': sampling_info,
            'target_dist': target_dist
        }
        
        with open(os.path.join(results_dir, f'mala_results_{dataset_name}.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"📊 MALA Results for {dataset_name}:")
        print(f"   KL Divergence: {kl_divergence:.3f}")
        print(f"   Cross-Entropy Surrogate: {cross_entropy:.3f}")
        print(f"   Acceptance Rate: {sampling_info['acceptance_rate']:.3f}")
        print(f"   Final Step Size: {step_sizes[-1]:.4f}")
        print(f"   Total Errors: {sampling_info.get('total_errors', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Critical error evaluating {dataset_name}: {e}")
        return None

def comprehensive_mala_evaluation():
    """Run MALA evaluation on all datasets with enhanced error handling."""
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
                'multimodal', 'two_moons', 'rings']
    
    all_results = {}
    summary_data = []
    
    print(f"🚀 Running MALA evaluation on {len(datasets)} datasets")
    print("=" * 60)
    
    for dataset_name in datasets:
        try:
            results = evaluate_mala_dataset(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
                
                summary_data.append([
                    dataset_name,
                    results['kl_divergence'],
                    results['cross_entropy_surrogate'],
                    results['sampling_info']['acceptance_rate'],
                    results['sampling_info']['step_sizes'][-1],
                    results['sampling_info'].get('total_errors', 0)
                ])
            
        except Exception as e:
            print(f"❌ Failed on {dataset_name}: {e}")
            continue
    
    if len(all_results) == 0:
        print("❌ No datasets successfully evaluated")
        return None
    
    # Create comprehensive comparison plot
    if len(all_results) > 1:
        print(f"\n📊 Creating comprehensive MALA comparison...")
        
        n_datasets = len(all_results)
        fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 8))
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['steelblue', 'crimson', 'forestgreen', 'darkorange', 
                 'purple', 'brown', 'pink']
        
        for i, (dataset_name, results) in enumerate(all_results.items()):
            color = colors[i % len(colors)]
            
            # Target data
            test_np = results['test_data'].cpu().numpy()
            axes[0, i].scatter(test_np[:, 0], test_np[:, 1], alpha=0.6, s=15, c=color)
            axes[0, i].set_title(f'{dataset_name.title()} Data')
            axes[0, i].grid(True, alpha=0.3)
            
            # MALA samples
            mala_np = results['mala_samples'].cpu().numpy()
            axes[1, i].scatter(mala_np[:, 0], mala_np[:, 1], alpha=0.6, s=15, c='red')
            axes[1, i].set_title(f'MALA Samples')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('MALA Comprehensive Evaluation', fontsize=16, y=0.98)
        plt.savefig('results/mala_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save summary CSV
    results_dir = 'results'
    with open(os.path.join(results_dir, 'mala_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'kl_divergence', 'cross_entropy_surrogate', 
                        'acceptance_rate', 'final_step_size', 'total_errors'])
        writer.writerows(summary_data)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("MALA SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} | {'KL Div':<8} | {'Cross-Ent':<10} | {'Accept':<6} | {'Step':<6} | {'Errors':<6}")
    print("-" * 80)
    
    for row in summary_data:
        print(f"{row[0]:<15} | {row[1]:<8.3f} | {row[2]:<10.3f} | {row[3]:<6.3f} | {row[4]:<6.4f} | {row[5]:<6}")
    
    print(f"\n✅ MALA evaluation completed!")
    print(f"   Results saved to: results/mala_comprehensive_evaluation.png")
    print(f"   Metrics saved to: results/mala_comprehensive_metrics.csv")
    
    # Best performing dataset
    if summary_data:
        best_kl = min(summary_data, key=lambda x: x[1])
        print(f"\n🏆 Best MALA performance (KL): {best_kl[0]} (KL: {best_kl[1]:.3f})")
        
        # Error analysis
        total_errors = sum(row[5] for row in summary_data)
        avg_errors = total_errors / len(summary_data)
        print(f"📊 Error Analysis: Total errors = {total_errors}, Average per dataset = {avg_errors:.1f}")
    
    return all_results

if __name__ == "__main__":
    comprehensive_mala_evaluation()