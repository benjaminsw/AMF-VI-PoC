import torch
import numpy as np
from scipy.stats import entropy

def compute_kl_divergence(p_samples, q_samples, n_bins=50):
    """Compute KL divergence between two 2D sample sets."""
    # Create 2D histogram bins
    x_min = min(p_samples[:, 0].min(), q_samples[:, 0].min()) - 0.1
    x_max = max(p_samples[:, 0].max(), q_samples[:, 0].max()) + 0.1
    y_min = min(p_samples[:, 1].min(), q_samples[:, 1].min()) - 0.1
    y_max = max(p_samples[:, 1].max(), q_samples[:, 1].max()) + 0.1
    
    # Compute histograms
    p_hist, _, _ = np.histogram2d(p_samples[:, 0], p_samples[:, 1], 
                                  bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])
    q_hist, _, _ = np.histogram2d(q_samples[:, 0], q_samples[:, 1], 
                                  bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])
    
    # Normalize to probabilities
    p_hist = p_hist.flatten() + 1e-8
    q_hist = q_hist.flatten() + 1e-8
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    
    # Compute KL divergence
    kl_div = entropy(p_hist, q_hist)
    return kl_div

class WeightEvaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate_weight_dynamics(self, weight_history):
        """Evaluate weight learning dynamics."""
        if not weight_history:
            return {}
        
        # Weight entropy over time
        weight_entropies = []
        for weights in weight_history:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            ent = entropy(weights + 1e-8)
            weight_entropies.append(ent)
        
        return {
            'final_entropy': weight_entropies[-1] if weight_entropies else 0,
            'entropy_evolution': weight_entropies
        }