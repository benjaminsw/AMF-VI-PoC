import math
from abc import ABC, abstractmethod

class BaseUpdateScheduler(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def get_learning_rate(self, epoch, total_epochs):
        pass

class ExponentialDecayScheduler(BaseUpdateScheduler):
    def get_learning_rate(self, epoch, total_epochs):
        initial_lr = self.config.scheduler_params.get('initial_lr', 0.01)
        decay_rate = self.config.scheduler_params.get('decay_rate', 0.95)
        return initial_lr * (decay_rate ** epoch)

class UpdateSchedulerFactory:
    @staticmethod
    def create_scheduler(config):
        if config.scheduler == 'exponential_decay':
            return ExponentialDecayScheduler(config)
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")