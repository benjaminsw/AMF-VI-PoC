import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class WeightLearningConfig:
    method: str = 'log_likelihood'
    strategy: str = 'direct_likelihood'
    scheduler: str = 'exponential_decay'
    likelihood_params: Dict[str, Any] = field(default_factory=dict)
    gradient_params: Dict[str, Any] = field(default_factory=dict)
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)

def load_config(config_path: str) -> WeightLearningConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return WeightLearningConfig(**config_dict)