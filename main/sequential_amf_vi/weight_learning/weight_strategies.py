from abc import ABC, abstractmethod

class BaseWeightStrategy(ABC):
    def __init__(self, config):
        self.config = config

class DirectLikelihoodStrategy(BaseWeightStrategy):
    def compute_weight_update(self, flows, data, current_weights, **kwargs):
        """Direct likelihood comparison strategy."""
        return current_weights  # Minimal implementation

class WeightStrategyFactory:
    @staticmethod
    def create_strategy(config):
        if config.strategy == 'direct_likelihood':
            return DirectLikelihoodStrategy(config)
        else:
            raise ValueError(f"Unknown weight strategy: {config.strategy}")