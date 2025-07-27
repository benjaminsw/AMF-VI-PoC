import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseWeightLearner(ABC):
    def __init__(self, config):
        self.config = config
        self.weight_history = []
    
    @abstractmethod
    def train_weights(self, model, data, strategy, scheduler):
        pass

class LogLikelihoodWeightLearner(BaseWeightLearner):
    def train_weights(self, model, data, strategy, scheduler):
        """Log-likelihood based weight learning."""
        epochs = self.config.training_params.get('weight_epochs', 300)
        weight_losses = []
        
        optimizer = torch.optim.Adam([model.log_weights], lr=scheduler.get_learning_rate(0, epochs))
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Get flow log probabilities
            flow_log_probs = []
            for flow in model.flows:
                flow.eval()
                with torch.no_grad():
                    log_prob = flow.log_prob(data)
                    flow_log_probs.append(log_prob.unsqueeze(1))
            
            flow_predictions = torch.cat(flow_log_probs, dim=1)
            
            # Compute mixture loss
            weights = F.softmax(model.log_weights, dim=0)
            batch_weights = weights.unsqueeze(0).expand(data.size(0), -1)
            weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            loss = -mixture_log_prob.mean()
            
            loss.backward()
            optimizer.step()
            weight_losses.append(loss.item())
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.get_learning_rate(epoch, epochs)
        
        return weight_losses

class WeightLearnerFactory:
    @staticmethod
    def create_learner(config):
        if config.method == 'log_likelihood':
            return LogLikelihoodWeightLearner(config)
        else:
            raise ValueError(f"Unknown weight learning method: {config.method}")