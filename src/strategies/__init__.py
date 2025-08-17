"""
Base classes and utilities for training strategies
"""

from abc import ABC, abstractmethod
import numpy as np
import yaml
import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrainingStats:
    """Statistics used for training decisions"""
    episode: int
    avg_score: float
    survival_rate: float
    avg_reward: float
    recent_episodes: int = 100
    
    def __post_init__(self):
        """Ensure all values are valid"""
        self.avg_score = max(0.0, self.avg_score)
        self.survival_rate = max(0.0, min(1.0, self.survival_rate))


@dataclass
class TrainingContext:
    """Context information for training strategies"""
    episode: int
    stats: TrainingStats
    reward_weights: Dict[str, float]
    config: Dict[str, Any]
    

class TrainingStrategy(ABC):
    """Base class for training strategies"""
    
    def __init__(self, name: str, description: str, config_name: Optional[str] = None):
        self.name = name
        self.description = description
        self.config_name = config_name or f"{name}_config.yaml"
        self.config = self._load_config()
        self.reward_weights = self.config.get('reward_weights', {})
        self.strategy_params = self.config.get('strategy_params', {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load strategy-specific configuration"""
        config_path = os.path.join(os.path.dirname(__file__), 'configs', self.config_name)
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_name} not found, using defaults")
            return {}
        except Exception as e:
            print(f"Warning: Error loading config {self.config_name}: {e}, using defaults")
            return {}
    
    def get_reward_weights(self) -> Dict[str, float]:
        """Get strategy-specific reward weights"""
        # Use custom weights if available, otherwise use config weights
        if hasattr(self, '_custom_weights') and self._custom_weights:
            return self._custom_weights.copy()
        return self.reward_weights.copy()
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy-specific parameters"""
        return self.strategy_params.copy()
        
    @abstractmethod
    def should_activate(self, context: TrainingContext) -> bool:
        """Determine if this strategy should be active for current context"""
        pass
        
    @abstractmethod
    def calculate_reward(
        self, 
        state: np.ndarray, 
        next_state: np.ndarray, 
        wall_collision: bool, 
        self_collision: bool, 
        ate_food: bool,
        context: TrainingContext,
        game_instance=None
    ) -> float:
        """Calculate reward for this strategy
        
        Args:
            state: Feature vector representing current state
            next_state: Feature vector representing next state
            wall_collision: Whether a wall collision occurred
            self_collision: Whether a self collision occurred
            ate_food: Whether food was eaten
            context: Training context with episode and stats info
            game_instance: Optional game instance for accessing full game state
        """
        pass
        
    @abstractmethod
    def get_stage_name(self) -> str:
        """Get display name for this training stage"""
        pass
        
    def modify_epsilon(self, base_epsilon: float, context: TrainingContext) -> float:
        """Modify exploration rate for this strategy (optional override)"""
        return base_epsilon
        
    def get_max_steps(self, base_steps: int, extra_per_food: int, score: int) -> int:
        """Get max steps for this strategy (optional override)"""
        return base_steps + (extra_per_food * score)
