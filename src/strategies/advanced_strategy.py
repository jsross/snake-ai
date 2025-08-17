"""
Advanced Training Strategy

Focus on optimal play with enhanced food emphasis and stronger guidance.
This strategy is for high-performing models that have mastered basic skills.
"""

import numpy as np
from . import TrainingStrategy, TrainingContext


class AdvancedTrainingStrategy(TrainingStrategy):
    """Advanced training strategy for optimal play"""
    
    def __init__(self):
        super().__init__(
            name="advanced",
            description="Optimize for high scores and efficient play",
            config_name="advanced_config.yaml"
        )
        
    def should_activate(self, context: TrainingContext) -> bool:
        """Activate for advanced play"""
        params = self.get_strategy_params()
        threshold = params.get('activation_threshold', {})
        
        return (context.episode >= threshold.get('min_episode', 2000) and 
                context.stats.survival_rate >= threshold.get('min_survival_rate', 0.8) and
                context.stats.avg_score >= threshold.get('min_avg_score', 3.0))
                
    def get_stage_name(self) -> str:
        return "Advanced Play"
        
    def get_stage_emoji(self) -> str:
        return "🎯"
        
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
        """Advanced reward calculation with enhanced food emphasis"""
        weights = self.get_reward_weights()
        
        # Death penalties from config
        if wall_collision:
            return weights.get('wall_collision_penalty', -50.0)
            
        if self_collision:
            return weights.get('self_collision_penalty', -60.0)
            
        # Enhanced food reward from config
        if ate_food:
            return weights.get('food_reward', 150.0)
            
        # Base reward
        reward = weights.get('survival_reward', 0.1) + weights.get('move_cost', -0.01)
        
        # Enhanced distance-based guidance using features and game instance
        if game_instance is not None and len(state) >= 5 and len(next_state) >= 5:
            # Use the normalized food distance from features (more efficient)
            prev_food_distance = state[4]  # normalized food distance
            new_food_distance = next_state[4]
            
            if new_food_distance < prev_food_distance:
                reward += weights.get('closer_reward', 0.75)
            elif new_food_distance > prev_food_distance:
                reward += weights.get('farther_penalty', -0.12)
                
            # Advanced bonus: reward staying in open areas (using available space features)
            # Features 13-16 are space counts in each direction
            if len(next_state) >= 17:
                avg_space = sum(next_state[13:17]) / 4  # Average available space
                reward += avg_space * weights.get('space_bonus', 0.1)
            
        return reward
        
    def modify_epsilon(self, base_epsilon: float, context: TrainingContext) -> float:
        """Lower exploration in advanced stage"""
        multiplier = self.strategy_params.get('exploration_multiplier', 0.8)
        return base_epsilon * multiplier
