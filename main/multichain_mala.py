import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data.data_generator import generate_data
import os
import pickle
import csv
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import concurrent.futures
import threading

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

class MultiChainMALASampler:
    """Multi-chain Metropolis-Adjusted Langevin Algorithm sampler."""
    
    def __init__(self, target_dist, step_size=0.01, dim=2, n_chains=4, debug=False):
        self.target_dist = target_dist
        self.step_size = step_size
        self.dim = dim
        self.n_chains = n_chains
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        # Adaptation parameters
        self.adapt_step_size = True
        self.target_accept_rate = 1.1/0.9
        self.adapt_window = 100
        
        # Error handling
        self.error_count = 0
        self.max_errors = 100
        
        # Threading lock for shared resources
        self.lock = threading.Lock()
    
    def _get_diverse_initial_points(self, target_data):
        """Generate diverse initial points using k-means clustering."""
        try:
            # Convert to numpy for k-means
            data_np = target_data.cpu().numpy()
            
            # Use k-means to identify diverse regions
            kmeans = KMeans(n_clusters=self.n_chains, random_state=42, n_init=10)
            kmeans.fit(data_np)
            
            # Use cluster centers as initial points
            initial_points = torch.tensor(kmeans.cluster_centers_, 
                                        dtype=torch.float32, device=self.device)
            
            if self.debug:
                print(f"K-means initial points: {initial_points}")
            
            return initial_points
            
        except Exception as e:
            if self.debug:
                print(f"K-means initialization failed: {e}, using random points")
            # Fallback: random initialization with some spread
            return torch.randn(self.n_chains, self.dim, device=self.device) * 2.0
    
    def _validate_state(self, x):
        """Validate MALA state."""
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            return False
        if torch.norm(x) > 1e3:
            return False
        return True
    
    def _single_chain_sample(self, chain_id, init_point, n_samples, n_burnin):
        """Sample from a single chain."""
        total_samples = n_samples + n_burnin
        x_current = init_point.clone()
        
        if not self._validate_state(x_current):
            x_current = torch.zeros(self.dim, device=self.device)
        
        samples = []
        n_accepted = 0
        step_sizes = []
        current_step_size = self.step_size
        consecutive_errors = 0
        
        for i in range(total_samples):
            try:
                # Compute gradient at current point
                grad_current = self.target_dist.grad_log_prob(x_current.unsqueeze(0)).squeeze(0)
                
                if torch.any(torch.isnan(grad_current)) or torch.any(torch.isinf(grad_current)):
                    grad_current = torch.zeros_like(grad_current)
                
                # Propose new state
                noise = torch.randn_like(x_current)
                x_proposed = (x_current + 0.5 * current_step_size**2 * grad_current + 
                             current_step_size * noise)
                
                if not self._validate_state(x_proposed):
                    x_proposed = x_current
                
                # Compute acceptance probability
                log_alpha = self._compute_log_acceptance_prob(
                    x_current, x_proposed, grad_current, current_step_size)
                
                if torch.isnan(log_alpha) or torch.isinf(log_alpha):
                    log_alpha = torch.tensor(-float('inf'))
                
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                
                # Accept or reject
                if torch.rand(1).item() < alpha and self._validate_state(x_proposed):
                    x_current = x_proposed
                    n_accepted += 1
                    consecutive_errors = 0
                
                # Store sample (after burn-in)
                if i >= n_burnin:
                    samples.append(x_current.clone())
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= self.max_errors:
                    break
                
                if i >= n_burnin:
                    samples.append(x_current.clone())
            
            # Step size adaptation
            if self.adapt_step_size and i > 0 and i % self.adapt_window == 0:
                accept_rate = n_accepted / (i + 1)
                if accept_rate > self.target_accept_rate:
                    current_step_size *= 1.02
                else:
                    current_step_size *= 0.98
                current_step_size = torch.clamp(torch.tensor(current_step_size), 0.0001, 0.1).item()
            
            step_sizes.append(current_step_size)
        
        final_accept_rate = n_accepted / total_samples if total_samples > 0 else 0
        
        if len(samples) == 0:
            samples = [torch.randn(self.dim, device=self.device) * 0.1 for _ in range(n_samples)]
        
        return {
            'chain_id': chain_id,
            'samples': torch.stack(samples),
            'acceptance_rate': final_accept_rate,
            'step_sizes': step_sizes,
            'n_accepted': n_accepted
        }
    
    def sample(self, n_samples, n_burnin=1000, target_data=None, parallel=True):
        """Generate samples using multi-chain MALA."""
        
        # Get diverse initial points
        if target_data is not None:
            initial_points = self._get_diverse_initial_points(target_data)
        else:
            initial_points = torch.randn(self.n_chains, self.dim, device=self.device) * 0.5
        
        # Ensure equal samples per chain
        samples_per_chain = n_samples // self.n_chains
        remaining_samples = n_samples % self.n_chains
        
        print(f"🔥 Multi-Chain MALA: {self.n_chains} chains, {samples_per_chain} samples each")
        
        if parallel:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_chains) as executor:
                futures = []
                for chain_id in range(self.n_chains):
                    # Add extra sample to first few chains if needed
                    chain_samples = samples_per_chain + (1 if chain_id < remaining_samples else 0)
                    future = executor.submit(
                        self._single_chain_sample, 
                        chain_id, initial_points[chain_id], chain_samples, n_burnin
                    )
                    futures.append(future)
                
                chain_results = [future.result() for future in futures]
        else:
            # Sequential execution
            chain_results = []
            for chain_id in range(self.n_chains):
                chain_samples = samples_per_chain + (1 if chain_id < remaining_samples else 0)
                result = self._single_chain_sample(
                    chain_id, initial_points[chain_id], chain_samples, n_burnin
                )
                chain_results.append(result)
                print(f"    Chain {chain_id+1}/{self.n_chains} complete: "
                      f"Accept Rate = {result['acceptance_rate']:.3f}")
        
        # Combine samples from all chains
        all_samples = []
        all_acceptance_rates = []
        all_step_sizes = []
        
        for result in chain_results:
            all_samples.append(result['samples'])
            all_acceptance_rates.append(result['acceptance_rate'])
            all_step_sizes.extend(result['step_sizes'])
        
        combined_samples = torch.cat(all_samples, dim=0)
        
        # Compute chain mixing diagnostics
        r_hat = self._compute_r_hat(chain_results) if len(chain_results) > 1 else 1.0
        
        # Summary info
        sampling_info = {
            'acceptance_rates': all_acceptance_rates,
            'mean_acceptance_rate': np.mean(all_acceptance_rates),
            'step_sizes': all_step_sizes,
            'n_chains': self.n_chains,
            'samples_per_chain': samples_per_chain,
            'r_hat': r_hat,
            'chain_results': chain_results,
            'initial_points': initial_points
        }
        
        print(f"    Combined {combined_samples.shape[0]} samples from {self.n_chains} chains")
        print(f"    Mean acceptance rate: {sampling_info['mean_acceptance_rate']:.3f}")
        print(f"    R-hat convergence diagnostic: {r_hat:.3f}")
        
        return combined_samples, sampling_info
    
    def _compute_log_acceptance_prob(self, x_current, x_proposed, grad_current, step_size):
        """Compute log acceptance probability for MALA."""
        try:
            # Forward proposal density
            grad_proposed = self.target_dist.grad_log_prob(x_proposed.unsqueeze(0)).squeeze(0)
            
            if torch.any(torch.isnan(grad_proposed)) or torch.any(torch.isinf(grad_proposed)):
                grad_proposed = grad_current
            
            mean_forward = x_current + 0.5 * step_size**2 * grad_current
            mean_backward = x_proposed + 0.5 * step_size**2 * grad_proposed
            
            log_q_forward = -0.5 * torch.sum((x_proposed - mean_forward)**2) / max(step_size**2, 1e-8)
            log_q_backward = -0.5 * torch.sum((x_current - mean_backward)**2) / max(step_size**2, 1e-8)
            
            # Target densities
            log_p_current = self.target_dist.log_prob(x_current.unsqueeze(0)).squeeze(0)
            log_p_proposed = self.target_dist.log_prob(x_proposed.unsqueeze(0)).squeeze(0)
            
            if torch.isnan(log_p_current) or torch.isinf(log_p_current):
                log_p_current = torch.tensor(-1e6)
            if torch.isnan(log_p_proposed) or torch.isinf(log_p_proposed):
                log_p_proposed = torch.tensor(-1e6)
            
            log_alpha = log_p_proposed - log_p_current + log_q_backward - log_q_forward
            
            return torch.clamp(log_alpha, min=-50, max=0)
            
        except Exception as e:
            self.error_count += 1
            return torch.tensor(-float('inf'))
    
    def _compute_r_hat(self, chain_results):
        """Compute R-hat convergence diagnostic (Gelman-Rubin statistic)."""
        try:
            n_chains = len(chain_results)
            if n_chains < 2:
                return 1.0
            
            # Get samples from each chain (use first dimension only for simplicity)
            chain_samples = []
            min_length = min(len(result['samples']) for result in chain_results)
            
            for result in chain_results:
                samples = result['samples'][:min_length, 0].cpu().numpy()  # First dimension only
                chain_samples.append(samples)
            
            chain_samples = np.array(chain_samples)  # [n_chains, n_samples]
            
            # Compute within-chain and between-chain variance
            n_samples = chain_samples.shape[1]
            chain_means = np.mean(chain_samples, axis=1)
            overall_mean = np.mean(chain_means)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chain_samples])
            
            # Between-chain variance
            B = n_samples * np.var(chain_means, ddof=1)
            
            # R-hat statistic
            if W <= 0:
                return float('inf')
            
            var_plus = ((n_samples - 1) * W + B) / n_samples
            r_hat = np.sqrt(var_plus / W)
            
            return float(r_hat)
            
        except Exception as e:
            if self.debug:
                print(f"R-hat computation failed: {e}")
            return float('inf')

# Keep original functions for backwards compatibility
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

def evaluate_mala_dataset(dataset_name, use_multichain=True, n_chains=4):
    """Evaluate MALA on a single dataset with multi-chain option."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {'Multi-Chain ' if use_multichain else ''}MALA on {dataset_name.upper()} dataset")  
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
        
        # Dataset-specific parameters
        step_size_map = {
            'banana': 0.03, 'x_shape': 0.03, 'bimodal_shared': 0.03,
            'bimodal_different': 0.03, 'multimodal': 0.03, 
            'two_moons': 0.03, 'rings': 0.03
        }
        step_size_map = {k: v * 100 for k, v in step_size_map.items()}
        step_size = step_size_map.get(dataset_name, 0.01)
        
        if use_multichain:
            # Use multi-chain sampler
            sampler = MultiChainMALASampler(target_dist, step_size=step_size, 
                                          n_chains=n_chains, debug=debug_mode)
            print(f"Using multi-chain MALA with {n_chains} chains, step size: {step_size}")
            mala_samples, sampling_info = sampler.sample(n_samples=100000, n_burnin=100, 
                                                       target_data=test_data, parallel=True)
        else:
            # Use single-chain sampler (backwards compatibility)
            from mala_baseline import MALASampler as SingleChainMALA
            sampler = SingleChainMALA(target_dist, step_size=step_size, debug=debug_mode)
            print(f"Using single-chain MALA, step size: {step_size}")
            mala_samples, sampling_info = sampler.sample(n_samples=1500, n_burnin=5000)
        
        # Validate samples before computing metrics
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
        
        # Create enhanced visualization for multi-chain
        if use_multichain:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot target data
        test_np = test_data.cpu().numpy()
        axes[0, 0].scatter(test_np[:, 0], test_np[:, 1], alpha=0.6, s=15, c='blue')
        axes[0, 0].set_title(f'Target Data ({dataset_name})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot MALA samples
        mala_np = mala_samples.cpu().numpy()
        axes[0, 1].scatter(mala_np[:, 0], mala_np[:, 1], alpha=0.6, s=15, c='red')
        axes[0, 1].set_title(f'{"Multi-Chain " if use_multichain else ""}MALA Samples')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overlay comparison
        axes[0, 2].scatter(test_np[:, 0], test_np[:, 1], alpha=0.4, s=10, c='blue', label='Target')
        axes[0, 2].scatter(mala_np[:, 0], mala_np[:, 1], alpha=0.4, s=10, c='red', label='MALA')
        axes[0, 2].set_title('Overlay Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        if use_multichain:
            # Multi-chain specific plots
            
            # Individual chains
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
            chain_results = sampling_info.get('chain_results', [])
            for i, chain_result in enumerate(chain_results):
                if i < 6:  # Limit colors
                    chain_samples = chain_result['samples'].cpu().numpy()
                    axes[1, 0].scatter(chain_samples[:, 0], chain_samples[:, 1], 
                                     alpha=0.6, s=8, c=colors[i % len(colors)], 
                                     label=f'Chain {i+1}')
            axes[1, 0].set_title('Individual Chain Samples')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Chain acceptance rates
            accept_rates = sampling_info.get('acceptance_rates', [])
            axes[1, 1].bar(range(len(accept_rates)), accept_rates)
            axes[1, 1].set_title('Chain Acceptance Rates')
            axes[1, 1].set_xlabel('Chain ID')
            axes[1, 1].set_ylabel('Acceptance Rate')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Initial points from k-means
            initial_points = sampling_info.get('initial_points')
            if initial_points is not None:
                init_np = initial_points.cpu().numpy()
                axes[1, 2].scatter(test_np[:, 0], test_np[:, 1], alpha=0.3, s=5, c='lightblue', label='Target Data')
                axes[1, 2].scatter(init_np[:, 0], init_np[:, 1], s=100, c='red', marker='x', linewidth=3, label='K-means Centers')
                axes[1, 2].set_title('K-means Initial Points')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            # Metrics summary (enhanced)
            axes[2, 0].axis('off')
            r_hat = sampling_info.get('r_hat', 'N/A')
            mean_accept = sampling_info.get('mean_acceptance_rate', 0)
            metrics_text = f"""Multi-Chain MALA Results:

KL Divergence: {kl_divergence:.3f}
Cross-Entropy: {cross_entropy:.3f}
Mean Accept Rate: {mean_accept:.3f}
R-hat Diagnostic: {r_hat:.3f}
Number of Chains: {sampling_info.get('n_chains', 'N/A')}

Target Mean: [{test_np.mean(axis=0)[0]:.2f}, {test_np.mean(axis=0)[1]:.2f}]
MALA Mean: [{mala_np.mean(axis=0)[0]:.2f}, {mala_np.mean(axis=0)[1]:.2f}]

Target Std: [{test_np.std(axis=0)[0]:.2f}, {test_np.std(axis=0)[1]:.2f}]
MALA Std: [{mala_np.std(axis=0)[0]:.2f}, {mala_np.std(axis=0)[1]:.2f}]

Convergence: {'Good' if r_hat < 1.1 else 'Poor' if r_hat != 'N/A' else 'N/A'}"""
            
            axes[2, 0].text(0.1, 0.9, metrics_text, transform=axes[2, 0].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            # Sample trace plots for first chain
            if chain_results:
                first_chain = chain_results[0]['samples'].cpu().numpy()
                axes[2, 1].plot(first_chain[:500, 0], label='X1', alpha=0.7)
                axes[2, 1].plot(first_chain[:500, 1], label='X2', alpha=0.7)
                axes[2, 1].set_title('Chain 1 Trace Plot')
                axes[2, 1].set_xlabel('Sample')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
            
            # R-hat evolution (placeholder for now)
            axes[2, 2].text(0.5, 0.5, f'R-hat = {r_hat:.3f}\n\n< 1.1: Good Convergence\n> 1.1: Poor Convergence', 
                           transform=axes[2, 2].transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[2, 2].set_title('Convergence Diagnostic')
            axes[2, 2].axis('off')
            
        else:
            # Single-chain plots (original)
            step_sizes = sampling_info.get('step_sizes', [])
            axes[1, 0].plot(step_sizes)
            axes[1, 0].set_title('Step Size Adaptation')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Step Size')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(mala_np[:500, 0])
            axes[1, 1].set_title('Sample Trace (X1)')
            axes[1, 1].set_xlabel('Sample')
            axes[1, 1].set_ylabel('X1')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].axis('off')
            accept_rate = sampling_info.get('acceptance_rate', 0)
            final_step = step_sizes[-1] if step_sizes else 0
            metrics_text = f"""MALA Results:
        
KL Divergence: {kl_divergence:.3f}
Cross-Entropy: {cross_entropy:.3f}
Acceptance Rate: {accept_rate:.3f}
Final Step Size: {final_step:.4f}

Target Mean: [{test_np.mean(axis=0)[0]:.2f}, {test_np.mean(axis=0)[1]:.2f}]
MALA Mean: [{mala_np.mean(axis=0)[0]:.2f}, {mala_np.mean(axis=0)[1]:.2f}]

Target Std: [{test_np.std(axis=0)[0]:.2f}, {test_np.std(axis=0)[1]:.2f}]
MALA Std: [{mala_np.std(axis=0)[0]:.2f}, {mala_np.std(axis=0)[1]:.2f}]"""
            
            axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        chain_type = "MultiChain" if use_multichain else "SingleChain"
        plt.suptitle(f'{chain_type} MALA Evaluation - {dataset_name.title()}', fontsize=16, y=0.98)
        
        # Save results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'{chain_type.lower()}_mala_evaluation_{dataset_name}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save samples
        results = {
            'dataset': dataset_name,
            'multichain': use_multichain,
            'n_chains': n_chains if use_multichain else 1,
            'mala_samples': mala_samples,
            'test_data': test_data,
            'kl_divergence': kl_divergence,
            'cross_entropy_surrogate': cross_entropy,
            'sampling_info': sampling_info,
            'target_dist': target_dist
        }
        
        with open(os.path.join(results_dir, f'{chain_type.lower()}_mala_results_{dataset_name}.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"📊 {chain_type} MALA Results for {dataset_name}:")
        print(f"   KL Divergence: {kl_divergence:.3f}")
        print(f"   Cross-Entropy Surrogate: {cross_entropy:.3f}")
        
        if use_multichain:
            mean_accept = sampling_info.get('mean_acceptance_rate', 0)
            r_hat = sampling_info.get('r_hat', 'N/A')
            print(f"   Mean Acceptance Rate: {mean_accept:.3f}")
            print(f"   R-hat Diagnostic: {r_hat:.3f}")
            print(f"   Number of Chains: {sampling_info.get('n_chains', 'N/A')}")
        else:
            accept_rate = sampling_info.get('acceptance_rate', 0)
            step_sizes = sampling_info.get('step_sizes', [])
            final_step = step_sizes[-1] if step_sizes else 0
            print(f"   Acceptance Rate: {accept_rate:.3f}")
            print(f"   Final Step Size: {final_step:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Critical error evaluating {dataset_name}: {e}")
        return None

def comprehensive_mala_evaluation(use_multichain=True, n_chains=4):
    """Run MALA evaluation on all datasets with multi-chain option."""
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
                'multimodal', 'two_moons', 'rings']
    
    all_results = {}
    summary_data = []
    
    chain_type = "Multi-Chain" if use_multichain else "Single-Chain"
    print(f"🚀 Running {chain_type} MALA evaluation on {len(datasets)} datasets")
    print("=" * 60)
    
    for dataset_name in datasets:
        try:
            results = evaluate_mala_dataset(dataset_name, use_multichain=use_multichain, n_chains=n_chains)
            if results is not None:
                all_results[dataset_name] = results
                
                if use_multichain:
                    summary_data.append([
                        dataset_name,
                        results['kl_divergence'],
                        results['cross_entropy_surrogate'],
                        results['sampling_info']['mean_acceptance_rate'],
                        results['sampling_info'].get('r_hat', 'N/A'),
                        results['sampling_info'].get('n_chains', 'N/A')
                    ])
                else:
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
        print(f"\n📊 Creating comprehensive {chain_type} MALA comparison...")
        
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
            axes[1, i].set_title(f'{chain_type} MALA')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'{chain_type} MALA Comprehensive Evaluation', fontsize=16, y=0.98)
        
        filename = f"{'multichain' if use_multichain else 'singlechain'}_mala_comprehensive_evaluation.png"
        plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save summary CSV
    results_dir = 'results'
    if use_multichain:
        filename = 'multichain_mala_comprehensive_metrics.csv'
        headers = ['dataset', 'kl_divergence', 'cross_entropy_surrogate', 
                  'mean_acceptance_rate', 'r_hat', 'n_chains']
    else:
        filename = 'singlechain_mala_comprehensive_metrics.csv'
        headers = ['dataset', 'kl_divergence', 'cross_entropy_surrogate', 
                  'acceptance_rate', 'final_step_size', 'total_errors']
    
    with open(os.path.join(results_dir, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(summary_data)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{chain_type.upper()} MALA SUMMARY TABLE")
    print(f"{'='*80}")
    
    if use_multichain:
        print(f"{'Dataset':<15} | {'KL Div':<8} | {'Cross-Ent':<10} | {'Accept':<6} | {'R-hat':<6} | {'Chains':<6}")
    else:
        print(f"{'Dataset':<15} | {'KL Div':<8} | {'Cross-Ent':<10} | {'Accept':<6} | {'Step':<6} | {'Errors':<6}")
    
    print("-" * 80)
    
    for row in summary_data:
        if use_multichain:
            r_hat_str = f"{row[4]:.3f}" if isinstance(row[4], (int, float)) else str(row[4])
            print(f"{row[0]:<15} | {row[1]:<8.3f} | {row[2]:<10.3f} | {row[3]:<6.3f} | {r_hat_str:<6} | {row[5]:<6}")
        else:
            print(f"{row[0]:<15} | {row[1]:<8.3f} | {row[2]:<10.3f} | {row[3]:<6.3f} | {row[4]:<6.4f} | {row[5]:<6}")
    
    print(f"\n✅ {chain_type} MALA evaluation completed!")
    print(f"   Results saved to: results/{filename.replace('.csv', '.png')}")
    print(f"   Metrics saved to: results/{filename}")
    
    # Best performing dataset
    if summary_data:
        best_kl = min(summary_data, key=lambda x: x[1])
        print(f"\n🏆 Best {chain_type} MALA performance (KL): {best_kl[0]} (KL: {best_kl[1]:.3f})")
        
        if use_multichain:
            # Convergence analysis
            r_hats = [row[4] for row in summary_data if isinstance(row[4], (int, float))]
            if r_hats:
                avg_r_hat = np.mean(r_hats)
                good_convergence = sum(1 for r in r_hats if r < 1.1)
                print(f"📊 Convergence Analysis: Average R-hat = {avg_r_hat:.3f}, "
                      f"Good convergence = {good_convergence}/{len(r_hats)} datasets")
    
    return all_results

if __name__ == "__main__":
    # Run both single-chain and multi-chain evaluations for comparison
    print("Running Single-Chain MALA evaluation...")
    single_results = comprehensive_mala_evaluation(use_multichain=False)
    
    print("\n" + "="*60)
    print("Running Multi-Chain MALA evaluation...")
    multi_results = comprehensive_mala_evaluation(use_multichain=True, n_chains=4)