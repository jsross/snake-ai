"""
Example Custom Strategy

This demonstrates how easy it is to add new training strategies
to the modular framework.
"""

import numpy as np
from . import TrainingStrategy, TrainingContext


class ExampleCustomStrategy(TrainingStrategy):
    """
    Example custom strategy that focuses on wall avoidance
    """
    
    def __init__(self):
        super().__init__(
            name="example_custom",
            description="Example strategy focusing on wall avoidance"
        )
    
    def should_activate(self, context: TrainingContext) -> bool:
        """Activate when we want to test custom behavior"""
        # For demo purposes, activate after 1000 episodes
        return context.episode > 1000
    
    def calculate_reward(self, context: TrainingContext) -> float:
        """Custom reward calculation with heavy wall penalty"""
        reward = 0.0
        
        # Heavy penalty for hitting walls
        if context.done and context.reason == "wall":
            reward -= 50.0  # Much higher than normal
        
        # Normal penalties for other deaths
        elif context.done:
            reward -= 10.0
            
        # Small bonus for staying alive
        else:
            reward += 0.5
            
        # Small bonus for food
        if context.food_eaten:
            reward += 5.0
            
        return reward
    
    def get_stage_name(self) -> str:
        return "Example Custom Training"
    
    def get_exploration_rate(self, base_rate: float) -> float:
        """Slightly higher exploration for testing"""
        return base_rate * 1.1
