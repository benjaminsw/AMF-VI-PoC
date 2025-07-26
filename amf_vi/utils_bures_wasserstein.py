"""
Utility functions for Gaussian Mixture Posterior with Bures-Wasserstein optimization.
"""

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Try to import optional dependencies
try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False

try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

def fit_gaussian(samples, reg_covar=1e-6):
    """
    Fit a single Gaussian to samples.
    
    Args:
        samples: torch.Tensor [n_samples, dim]
        reg_covar: float - regularization for covariance matrix
    
    Returns:
        dict with 'mean' and 'cov' tensors
    """
    if samples.size(0) == 0:
        raise ValueError("Cannot fit Gaussian to empty samples")
    
    mean = samples.mean(dim=0)
    
    if samples.size(0) == 1:
        # Single sample case
        cov = torch.eye(samples.size(1), device=samples.device) * reg_covar
    else:
        # Compute covariance with regularization
        cov = torch.cov(samples.T) + reg_covar * torch.eye(samples.size(1), device=samples.device)
    
    return {'mean': mean, 'cov': cov}

def fit_gaussian_mixture(samples, n_components=5, method='gmm', reg_covar=1e-6):
    """
    Fit a Gaussian mixture model to samples.
    
    Args:
        samples: torch.Tensor [n_samples, dim]
        n_components: int - number of mixture components
        method: str - 'gmm' (EM) or 'kmeans' (K-means + local fitting)
        reg_covar: float - regularization for covariance matrices
    
    Returns:
        dict with 'means', 'covs', 'weights' tensors
    """
    device = samples.device
    dim = samples.size(1)
    n_samples = samples.size(0)
    
    if n_samples < n_components:
        # Not enough samples, return single component
        gaussian = fit_gaussian(samples, reg_covar)
        return {
            'means': gaussian['mean'].unsqueeze(0),
            'covs': gaussian['cov'].unsqueeze(0),
            'weights': torch.ones(1, device=device)
        }
    
    if method == 'gmm' and n_samples >= 2 * n_components:
        # Use sklearn's Gaussian Mixture Model
        samples_np = samples.detach().cpu().numpy()
        gmm = GaussianMixture(
            n_components=n_components, 
            random_state=42,
            reg_covar=reg_covar,
            max_iter=100,
            n_init=3
        )
        try:
            gmm.fit(samples_np)
            
            means = torch.tensor(gmm.means_, device=device, dtype=samples.dtype)
            covs = torch.tensor(gmm.covariances_, device=device, dtype=samples.dtype)
            weights = torch.tensor(gmm.weights_, device=device, dtype=samples.dtype)
            
            return {'means': means, 'covs': covs, 'weights': weights}
            
        except Exception as e:
            print(f"GMM fitting failed: {e}, falling back to K-means")
            method = 'kmeans'
    
    if method == 'kmeans':
        # Use K-means clustering + local Gaussian fitting
        samples_np = samples.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
        labels = kmeans.fit_predict(samples_np)
        
        means = []
        covs = []
        weights = []
        
        for i in range(n_components):
            mask = labels == i
            cluster_samples = samples[mask]
            
            if cluster_samples.size(0) > 0:
                gaussian = fit_gaussian(cluster_samples, reg_covar)
                means.append(gaussian['mean'])
                covs.append(gaussian['cov'])
                weights.append(float(mask.sum()) / n_samples)
            else:
                # Empty cluster, use random initialization
                means.append(samples[torch.randint(n_samples, (1,))].squeeze(0))
                covs.append(torch.eye(dim, device=device) * reg_covar)
                weights.append(0.0)
        
        means = torch.stack(means)
        covs = torch.stack(covs)
        weights = torch.tensor(weights, device=device, dtype=samples.dtype)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return {'means': means, 'covs': covs, 'weights': weights}
    
    else:
        # Fallback: single Gaussian
        gaussian = fit_gaussian(samples, reg_covar)
        return {
            'means': gaussian['mean'].unsqueeze(0),
            'covs': gaussian['cov'].unsqueeze(0),
            'weights': torch.ones(1, device=device)
        }

def update_covariance_bures_wasserstein(current_cov, target_cov, step_size=0.1, method='geoopt'):
    """
    Update covariance matrix using Bures-Wasserstein Riemannian optimization.
    
    Args:
        current_cov: torch.Tensor [dim, dim] - current covariance matrix
        target_cov: torch.Tensor [dim, dim] - target covariance matrix
        step_size: float - step size for update
        method: str - 'geoopt' (Riemannian) or 'euclidean' (fallback)
    
    Returns:
        torch.Tensor [dim, dim] - updated covariance matrix
    """
    device = current_cov.device
    dim = current_cov.size(0)
    
    if method == 'geoopt' and GEOOPT_AVAILABLE:
        # Use geoopt for proper Riemannian optimization on PSD manifold
        manifold = geoopt.SymmetricPositiveDefinite()
        
        with torch.no_grad():
            # Compute Riemannian gradient (simplified as Euclidean difference)
            grad = current_cov - target_cov
            
            # Project gradient onto tangent space and take exponential map step
            tangent_vec = manifold.proju(current_cov, -step_size * grad)
            new_cov = manifold.expmap(current_cov, tangent_vec)
            
            return new_cov
    
    else:
        # Fallback: Euclidean interpolation with PSD projection
        with torch.no_grad():
            # Linear interpolation
            interpolated = (1 - step_size) * current_cov + step_size * target_cov
            
            # Project back to PSD cone using eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(interpolated)
            eigenvals = torch.clamp(eigenvals, min=1e-6)  # Ensure positive definiteness
            projected_cov = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
            
            return projected_cov

def compute_bures_wasserstein_distance(cov1, cov2):
    """
    Compute Bures-Wasserstein distance between two covariance matrices.
    
    Args:
        cov1, cov2: torch.Tensor [dim, dim] - covariance matrices
    
    Returns:
        float - Bures-Wasserstein distance
    """
    # Bures-Wasserstein distance: ||C1 + C2 - 2(C1^{1/2} C2 C1^{1/2})^{1/2}||_F
    try:
        # Compute matrix square roots
        eigenvals1, eigenvecs1 = torch.linalg.eigh(cov1)
        eigenvals1 = torch.clamp(eigenvals1, min=1e-8)
        sqrt_cov1 = eigenvecs1 @ torch.diag(torch.sqrt(eigenvals1)) @ eigenvecs1.T
        
        # Compute the product sqrt(C1) * C2 * sqrt(C1)
        product = sqrt_cov1 @ cov2 @ sqrt_cov1
        eigenvals_prod, eigenvecs_prod = torch.linalg.eigh(product)
        eigenvals_prod = torch.clamp(eigenvals_prod, min=1e-8)
        sqrt_product = eigenvecs_prod @ torch.diag(torch.sqrt(eigenvals_prod)) @ eigenvecs_prod.T
        
        # Compute the distance
        distance_matrix = cov1 + cov2 - 2 * sqrt_product
        distance = torch.norm(distance_matrix, p='fro').item()
        
        return distance
    
    except Exception as e:
        # Fallback: Frobenius norm of difference
        return torch.norm(cov1 - cov2, p='fro').item()

def compute_wasserstein_barycenter(covariances, weights=None, max_iter=50, tol=1e-6):
    """
    Compute Wasserstein barycenter of covariance matrices using fixed-point iteration.
    
    Args:
        covariances: list of torch.Tensor [dim, dim] - covariance matrices
        weights: torch.Tensor [n_matrices] - barycenter weights (uniform if None)
        max_iter: int - maximum iterations
        tol: float - convergence tolerance
    
    Returns:
        torch.Tensor [dim, dim] - barycenter covariance matrix
    """
    if not covariances:
        raise ValueError("Need at least one covariance matrix")
    
    device = covariances[0].device
    n_matrices = len(covariances)
    
    if weights is None:
        weights = torch.ones(n_matrices, device=device) / n_matrices
    else:
        weights = weights / weights.sum()  # Normalize
    
    # Initialize barycenter as weighted average
    barycenter = torch.zeros_like(covariances[0])
    for cov, w in zip(covariances, weights):
        barycenter += w * cov
    
    # Fixed-point iteration for Wasserstein barycenter
    for iteration in range(max_iter):
        barycenter_old = barycenter.clone()
        
        try:
            # Compute matrix square root of current barycenter
            eigenvals, eigenvecs = torch.linalg.eigh(barycenter)
            eigenvals = torch.clamp(eigenvals, min=1e-8)
            sqrt_barycenter = eigenvecs @ torch.diag(torch.sqrt(eigenvals)) @ eigenvecs.T
            inv_sqrt_barycenter = eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
            
            # Update barycenter
            new_barycenter = torch.zeros_like(barycenter)
            for cov, w in zip(covariances, weights):
                # Compute (sqrt(B) * C_i * sqrt(B))^{1/2}
                product = sqrt_barycenter @ cov @ sqrt_barycenter
                eigenvals_prod, eigenvecs_prod = torch.linalg.eigh(product)
                eigenvals_prod = torch.clamp(eigenvals_prod, min=1e-8)
                sqrt_product = eigenvecs_prod @ torch.diag(torch.sqrt(eigenvals_prod)) @ eigenvecs_prod.T
                
                # Transform back: B^{-1/2} * (sqrt(B) * C_i * sqrt(B))^{1/2} * B^{-1/2}
                term = inv_sqrt_barycenter @ sqrt_product @ inv_sqrt_barycenter
                new_barycenter += w * term
            
            # Square the result
            new_barycenter = new_barycenter @ new_barycenter
            
            # Check convergence
            diff = torch.norm(new_barycenter - barycenter_old, p='fro')
            if diff < tol:
                break
            
            barycenter = new_barycenter
            
        except Exception as e:
            print(f"Barycenter iteration {iteration} failed: {e}, using weighted average")
            # Fallback to weighted average
            barycenter = torch.zeros_like(covariances[0])
            for cov, w in zip(covariances, weights):
                barycenter += w * cov
            break
    
    return barycenter

def compute_gaussian_mixture_log_prob(x, means, covs, weights, reg_covar=1e-6):
    """
    Compute log probability under a Gaussian mixture model.
    
    Args:
        x: torch.Tensor [batch_size, dim] - input points
        means: torch.Tensor [n_components, dim] - component means
        covs: torch.Tensor [n_components, dim, dim] - component covariances
        weights: torch.Tensor [n_components] - component weights
        reg_covar: float - regularization for numerical stability
    
    Returns:
        torch.Tensor [batch_size] - log probabilities
    """
    batch_size, dim = x.size()
    n_components = means.size(0)
    device = x.device
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Compute log probabilities for each component
    component_log_probs = torch.zeros(batch_size, n_components, device=device)
    
    for k in range(n_components):
        mean_k = means[k]  # [dim]
        cov_k = covs[k] + reg_covar * torch.eye(dim, device=device)  # [dim, dim]
        weight_k = weights[k]
        
        # Compute log probability for component k
        diff = x - mean_k.unsqueeze(0)  # [batch_size, dim]
        
        try:
            # Use Cholesky decomposition for efficiency
            chol = torch.linalg.cholesky(cov_k)
            log_det = 2 * torch.sum(torch.log(torch.diag(chol)))
            solve = torch.linalg.solve_triangular(chol, diff.T, upper=False)
            mahalanobis = torch.sum(solve ** 2, dim=0)
        except:
            # Fallback using eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(cov_k)
            eigenvals = torch.clamp(eigenvals, min=1e-8)
            log_det = torch.sum(torch.log(eigenvals))
            inv_cov = eigenvecs @ torch.diag(1.0 / eigenvals) @ eigenvecs.T
            mahalanobis = torch.sum(diff @ inv_cov * diff, dim=1)
        
        log_prob_k = (-0.5 * (dim * torch.log(2 * torch.tensor(np.pi, device=device)) + 
                             log_det + mahalanobis) + torch.log(weight_k))
        component_log_probs[:, k] = log_prob_k
    
    # Use logsumexp for numerical stability
    mixture_log_prob = torch.logsumexp(component_log_probs, dim=1)
    return mixture_log_prob

def sample_gaussian_mixture(n_samples, means, covs, weights, reg_covar=1e-6):
    """
    Sample from a Gaussian mixture model.
    
    Args:
        n_samples: int - number of samples
        means: torch.Tensor [n_components, dim] - component means
        covs: torch.Tensor [n_components, dim, dim] - component covariances
        weights: torch.Tensor [n_components] - component weights
        reg_covar: float - regularization for numerical stability
    
    Returns:
        torch.Tensor [n_samples, dim] - samples
    """
    device = means.device
    dim = means.size(1)
    n_components = means.size(0)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Sample component assignments
    component_samples = torch.multinomial(weights, n_samples, replacement=True)
    
    # Sample from assigned components
    samples = torch.zeros(n_samples, dim, device=device)
    
    for k in range(n_components):
        # Find samples assigned to component k
        mask = component_samples == k
        n_k = mask.sum().item()
        
        if n_k > 0:
            mean_k = means[k]
            cov_k = covs[k] + reg_covar * torch.eye(dim, device=device)
            
            # Generate Gaussian noise
            noise = torch.randn(n_k, dim, device=device)
            
            try:
                # Use Cholesky decomposition for sampling
                chol = torch.linalg.cholesky(cov_k)
                samples_k = mean_k.unsqueeze(0) + noise @ chol.T
            except:
                # Fallback using eigendecomposition
                eigenvals, eigenvecs = torch.linalg.eigh(cov_k)
                eigenvals = torch.clamp(eigenvals, min=1e-8)
                sqrt_cov = eigenvecs @ torch.diag(torch.sqrt(eigenvals)) @ eigenvecs.T
                samples_k = mean_k.unsqueeze(0) + noise @ sqrt_cov.T
            
            samples[mask] = samples_k
    
    return samples

def adaptive_covariance_regularization(cov, target_cov=None, reg_strength=1e-6, 
                                     condition_threshold=1e12):
    """
    Adaptively regularize covariance matrix based on condition number.
    
    Args:
        cov: torch.Tensor [dim, dim] - covariance matrix to regularize
        target_cov: torch.Tensor [dim, dim] - target covariance (optional)
        reg_strength: float - base regularization strength
        condition_threshold: float - condition number threshold for additional regularization
    
    Returns:
        torch.Tensor [dim, dim] - regularized covariance matrix
    """
    device = cov.device
    dim = cov.size(0)
    
    # Compute eigenvalues to check condition number
    eigenvals = torch.linalg.eigvals(cov).real
    eigenvals = torch.clamp(eigenvals, min=1e-12)
    condition_number = eigenvals.max() / eigenvals.min()
    
    # Base regularization
    regularized_cov = cov + reg_strength * torch.eye(dim, device=device)
    
    # Additional regularization if condition number is too high
    if condition_number > condition_threshold:
        additional_reg = (condition_number / condition_threshold) * reg_strength
        regularized_cov += additional_reg * torch.eye(dim, device=device)
    
    # If target covariance is provided, blend towards it for stability
    if target_cov is not None:
        target_condition = torch.linalg.eigvals(target_cov).real
        target_condition = torch.clamp(target_condition, min=1e-12)
        target_cond_num = target_condition.max() / target_condition.min()
        
        if target_cond_num < condition_number:
            # Blend towards better-conditioned target
            blend_factor = min(0.1, condition_number / (10 * condition_threshold))
            regularized_cov = (1 - blend_factor) * regularized_cov + blend_factor * target_cov
    
    return regularized_cov

def evaluate_gaussian_mixture_quality(samples, target_data, means, covs, weights):
    """
    Evaluate quality of Gaussian mixture approximation.
    
    Args:
        samples: torch.Tensor [n_samples, dim] - generated samples
        target_data: torch.Tensor [n_target, dim] - target data
        means: torch.Tensor [n_components, dim] - mixture means
        covs: torch.Tensor [n_components, dim, dim] - mixture covariances
        weights: torch.Tensor [n_components] - mixture weights
    
    Returns:
        dict with quality metrics
    """
    device = samples.device
    
    # Compute log-likelihood of target data under mixture
    target_log_likelihood = compute_gaussian_mixture_log_prob(target_data, means, covs, weights)
    mean_log_likelihood = target_log_likelihood.mean().item()
    
    # Compute coverage (fraction of target data within 2-sigma of any component)
    coverage_count = 0
    for i in range(target_data.size(0)):
        point = target_data[i:i+1]
        covered = False
        
        for k in range(means.size(0)):
            mean_k = means[k]
            cov_k = covs[k]
            
            # Compute Mahalanobis distance
            diff = point - mean_k.unsqueeze(0)
            try:
                inv_cov = torch.linalg.inv(cov_k + 1e-6 * torch.eye(cov_k.size(0), device=device))
                mahal_dist = torch.sqrt(diff @ inv_cov @ diff.T).item()
                if mahal_dist < 2.0:  # Within 2-sigma
                    covered = True
                    break
            except:
                continue
        
        if covered:
            coverage_count += 1
    
    coverage = coverage_count / target_data.size(0)
    
    # Compute sample quality using MMD
    def compute_mmd(X, Y, bandwidth=1.0):
        n, m = X.size(0), Y.size(0)
        X_sqnorms = torch.sum(X**2, dim=1, keepdim=True)
        Y_sqnorms = torch.sum(Y**2, dim=1, keepdim=True)
        
        XY = torch.mm(X, Y.t())
        XX = torch.mm(X, X.t())
        YY = torch.mm(Y, Y.t())
        
        X_dists = X_sqnorms - 2*XX + X_sqnorms.t()
        Y_dists = Y_sqnorms - 2*YY + Y_sqnorms.t()
        XY_dists = X_sqnorms - 2*XY + Y_sqnorms.t()
        
        K_XX = torch.exp(-X_dists / (2 * bandwidth**2))
        K_YY = torch.exp(-Y_dists / (2 * bandwidth**2))
        K_XY = torch.exp(-XY_dists / (2 * bandwidth**2))
        
        mmd_squared = K_XX.mean() + K_YY.mean() - 2*K_XY.mean()
        return torch.sqrt(torch.clamp(mmd_squared, min=0))
    
    mmd_score = compute_mmd(samples, target_data).item()
    
    # Compute effective number of components (entropy-based)
    normalized_weights = weights / weights.sum()
    entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8))
    effective_components = torch.exp(entropy).item()
    
    return {
        'mean_log_likelihood': mean_log_likelihood,
        'coverage': coverage,
        'mmd_score': mmd_score,
        'effective_components': effective_components,
        'n_components': len(weights),
        'weight_entropy': entropy.item()
    }

def visualize_gaussian_mixture_components(means, covs, weights, ax=None, colors=None, alpha=0.3):
    """
    Visualize Gaussian mixture components as ellipses (2D only).
    
    Args:
        means: torch.Tensor [n_components, 2] - component means
        covs: torch.Tensor [n_components, 2, 2] - component covariances
        weights: torch.Tensor [n_components] - component weights
        ax: matplotlib axis (optional)
        colors: list of colors (optional)
        alpha: float - transparency
    
    Returns:
        matplotlib axis with plotted ellipses
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    n_components = means.size(0)
    
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, n_components))
    
    # Normalize weights for visualization
    normalized_weights = weights / weights.sum()
    
    for k in range(n_components):
        mean = means[k].cpu().numpy()
        cov = covs[k].cpu().numpy()
        weight = normalized_weights[k].item()
        
        # Compute eigenvalues and eigenvectors for ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Ellipse dimensions (2-sigma)
        width, height = 2 * np.sqrt(eigenvals)
        
        # Scale by weight for visualization
        width *= np.sqrt(weight) * 2
        height *= np.sqrt(weight) * 2
        
        # Create ellipse
        ellipse = Ellipse(
            mean, width, height, angle=angle,
            facecolor=colors[k], edgecolor='black',
            alpha=alpha, linewidth=1.5
        )
        ax.add_patch(ellipse)
        
        # Add component label
        ax.text(mean[0], mean[1], f'C{k+1}\n{weight:.3f}', 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gaussian Mixture Components')
    
    return ax

def create_requirements_file():
    """Create requirements.txt file with necessary dependencies."""
    requirements = [
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.1.0",
        "geoopt>=0.5.0  # For Bures-Wasserstein Riemannian optimization",
        "POT>=0.8.0     # For Wasserstein barycenter computation"
    ]
    
    requirements_text = "\n".join(requirements)
    
    return requirements_text

# Example usage and testing functions
def test_bures_wasserstein_functionality():
    """Test the Bures-Wasserstein functionality with synthetic data."""
    print("Testing Bures-Wasserstein functionality...")
    
    # Create synthetic covariance matrices
    torch.manual_seed(42)
    dim = 3
    
    # Generate random PSD matrices
    A1 = torch.randn(dim, dim)
    cov1 = A1 @ A1.T + 0.1 * torch.eye(dim)
    
    A2 = torch.randn(dim, dim)
    cov2 = A2 @ A2.T + 0.1 * torch.eye(dim)
    
    print(f"Original covariances:")
    print(f"Cov1:\n{cov1}")
    print(f"Cov2:\n{cov2}")
    
    # Test Bures-Wasserstein distance
    distance = compute_bures_wasserstein_distance(cov1, cov2)
    print(f"Bures-Wasserstein distance: {distance:.4f}")
    
    # Test covariance update
    updated_cov = update_covariance_bures_wasserstein(cov1, cov2, step_size=0.5)
    print(f"Updated covariance:\n{updated_cov}")
    
    # Test barycenter computation
    covariances = [cov1, cov2]
    barycenter = compute_wasserstein_barycenter(covariances)
    print(f"Wasserstein barycenter:\n{barycenter}")
    
    # Test Gaussian fitting
    samples = torch.randn(100, dim)
    gaussian_fit = fit_gaussian(samples)
    print(f"Fitted Gaussian mean: {gaussian_fit['mean']}")
    print(f"Fitted Gaussian cov: {gaussian_fit['cov']}")
    
    # Test mixture fitting
    mixture_fit = fit_gaussian_mixture(samples, n_components=2)
    print(f"Mixture means shape: {mixture_fit['means'].shape}")
    print(f"Mixture covs shape: {mixture_fit['covs'].shape}")
    print(f"Mixture weights: {mixture_fit['weights']}")
    
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    # Run tests
    test_bures_wasserstein_functionality()
    
    # Print requirements
    print("\n" + "="*50)
    print("REQUIREMENTS.TXT")
    print("="*50)
    print(create_requirements_file())