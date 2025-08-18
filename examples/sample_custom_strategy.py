"""
Sample Custom Strategy

This is a template showing how to create a custom training strategy
in its own file within the strategies folder.
"""

import numpy as np
from . import TrainingStrategy, TrainingContext


class SampleCustomStrategy(TrainingStrategy):
    """Sample custom strategy to demonstrate the pattern"""
    
    def __init__(self):
        super().__init__(
            name="sample_custom",
            description="A sample strategy demonstrating custom implementation"
        )
        
    def should_activate(self, context: TrainingContext) -> bool:
        """Example activation logic - never actually activates"""
        # This strategy never activates by default
        # Change this logic to test your custom strategy
        return False
        
    def get_stage_name(self) -> str:
        return "Sample Custom"
        
    def calculate_reward(
        self, 
        state: np.ndarray, 
        next_state: np.ndarray, 
        wall_collision: bool, 
        self_collision: bool, 
        ate_food: bool,
        context: TrainingContext
    ) -> float:
        """Sample reward calculation"""
        weights = context.reward_weights
        
        # Standard death penalties
        if wall_collision:
            return weights['wall_collision_penalty']
            
        if self_collision:
            return weights['self_collision_penalty']
            
        # Standard food reward
        if ate_food:
            return weights['food_reward']
            
        # Custom logic: reward based on episode number
        base_reward = weights['survival_reward'] + weights['move_cost']
        
        # Example: slight bonus for later episodes
        episode_bonus = min(context.episode / 10000.0, 0.1)
        
        return base_reward + episode_bonus
        
    def modify_epsilon(self, base_epsilon: float, context: TrainingContext) -> float:
        """Example: gradually reduce exploration over time"""
        reduction_factor = min(context.episode / 5000.0, 0.5)
        return base_epsilon * (1.0 - reduction_factor)
        
    def get_max_steps(self, base_steps: int, extra_per_food: int, score: int) -> int:
        """Example: longer episodes for higher scores"""
        return base_steps + (extra_per_food * score) + min(score * 10, 100)
