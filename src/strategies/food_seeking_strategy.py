"""
Food Seeking Training Strategy

Focus on learning to move toward food while maintaining survival skills.
This strategy adds distance-based guidance after basic survival is mastered.
"""

import numpy as np
from . import TrainingStrategy, TrainingContext


class FoodSeekingTrainingStrategy(TrainingStrategy):
    """Training strategy focused on seeking food while maintaining survival"""
    
    def __init__(self):
        super().__init__(
            name="food_seeking",
            description="Learn to move toward food while avoiding death",
            config_name="food_seeking_config.yaml"
        )
        
    def should_activate(self, context: TrainingContext) -> bool:
        """Activate after survival is mastered but before advanced play"""
        params = self.get_strategy_params()
        threshold = params.get('activation_threshold', {})
        
        survival_mastered = (context.episode >= threshold.get('min_episode', 2000) and 
                           context.stats.survival_rate >= threshold.get('min_survival_rate', 0.8))
        not_advanced = context.stats.avg_score < threshold.get('max_avg_score', 3.0)
        return survival_mastered and not_advanced
        
    def get_stage_name(self) -> str:
        return "Food Seeking"
        
    def calculate_reward(
        self, 
        state: np.ndarray, 
        next_state: np.ndarray, 
        wall_collision: bool, 
        self_collision: bool, 
        ate_food: bool,
        context: TrainingContext
    ) -> float:
        """Reward calculation with distance-based guidance"""
        weights = self.get_reward_weights()
        
        # Death penalties from config
        if wall_collision:
            return weights.get('wall_collision_penalty', -50.0)
            
        if self_collision:
            return weights.get('self_collision_penalty', -60.0)
            
        # Food reward from config
        if ate_food:
            return weights.get('food_reward', 80.0)
            
        # Base reward
        reward = weights.get('survival_reward', 0.1) + weights.get('move_cost', -0.01)
        
        # Add distance-based guidance
        head_pos = np.argwhere(state == 1)[0]
        next_head_pos = np.argwhere(next_state == 1)[0]
        food_pos = np.argwhere(state < 0)[0]
        
        prev_distance = np.linalg.norm(head_pos - food_pos)
        new_distance = np.linalg.norm(next_head_pos - food_pos)
        
        if new_distance < prev_distance:
            reward += weights.get('closer_reward', 1.0)
        elif new_distance > prev_distance:
            reward += weights.get('farther_penalty', -0.2)
            
        return reward
        
    def modify_epsilon(self, base_epsilon: float, context: TrainingContext) -> float:
        """Slightly higher exploration for food seeking"""
        multiplier = self.strategy_params.get('exploration_multiplier', 1.1)
        return base_epsilon * multiplier
