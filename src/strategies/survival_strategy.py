"""
Survival Training Strategy

Focus on basic survival skills - avoiding walls and self-collision.
This is typically the first stage of training.
"""

import numpy as np
from . import TrainingStrategy, TrainingContext


class SurvivalTrainingStrategy(TrainingStrategy):
    """Training strategy focused on basic survival skills"""
    
    def __init__(self):
        super().__init__(
            name="survival",
            description="Focus on avoiding walls and self-collision",
            config_name="survival_config.yaml"
        )
        
    def should_activate(self, context: TrainingContext) -> bool:
        """Activate for early episodes or low survival rate"""
        params = self.get_strategy_params()
        threshold = params.get('activation_threshold', {})
        
        return (context.episode < threshold.get('min_episode', 2000) or 
                context.stats.survival_rate < threshold.get('max_survival_rate', 0.8))
                
    def get_stage_name(self) -> str:
        return "Survival Training"
        
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
        """Pure survival-focused reward calculation"""
        
        # Get reward weights from config
        weights = self.get_reward_weights()
        
        # Heavy penalties for death (survival focus only)
        if wall_collision:
            return weights.get('wall_collision_penalty')
            
        if self_collision:
            return weights.get('self_collision_penalty')
            
        # Small survival bonus for each step alive
        return weights.get('survival_reward', 0.2)
        
    def modify_epsilon(self, base_epsilon: float, context: TrainingContext) -> float:
        """Higher exploration for survival learning"""
        multiplier = self.strategy_params.get('exploration_multiplier', 1.2)
        return base_epsilon * multiplier
