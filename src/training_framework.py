"""
Training Framework for Snake AI

This module provides a pluggable framework for different training strategies.
Each stage of curriculum learning can use different training approaches.
"""

from typing import Tuple
import numpy as np

# Handle both relative and absolute imports
try:
    from .strategies import TrainingStrategy, TrainingContext, TrainingStats
    from .strategies.survival_strategy import SurvivalTrainingStrategy
    from .strategies.food_seeking_strategy import FoodSeekingTrainingStrategy
    from .strategies.advanced_strategy import AdvancedTrainingStrategy
except ImportError:
    from strategies import TrainingStrategy, TrainingContext, TrainingStats
    from strategies.survival_strategy import SurvivalTrainingStrategy
    from strategies.food_seeking_strategy import FoodSeekingTrainingStrategy
    from strategies.advanced_strategy import AdvancedTrainingStrategy


class TrainingFramework:
    """Framework for managing training strategies"""
    
    def __init__(self):
        self.strategies = [
            SurvivalTrainingStrategy(),
            FoodSeekingTrainingStrategy(),
            AdvancedTrainingStrategy()
        ]
        self.forced_strategy = None  # For manual strategy selection
        
    def set_strategy(self, strategy_name: str):
        """Force a specific strategy to be used"""
        if strategy_name == 'auto':
            self.forced_strategy = None
        else:
            strategy_map = {
                'survival': SurvivalTrainingStrategy,
                'food_seeking': FoodSeekingTrainingStrategy,
                'advanced': AdvancedTrainingStrategy
            }
            if strategy_name in strategy_map:
                self.forced_strategy = strategy_map[strategy_name]()
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        
    def get_active_strategy(self, context: TrainingContext) -> TrainingStrategy:
        """Get the appropriate strategy for current context"""
        # If a strategy is forced, use it
        if self.forced_strategy:
            return self.forced_strategy
            
        # Otherwise use automatic curriculum learning
        # Check strategies in reverse order (most advanced first)
        for strategy in reversed(self.strategies):
            if strategy.should_activate(context):
                return strategy
                
        # Fallback to first strategy (survival)
        return self.strategies[0]
        
    def calculate_reward(
        self, 
        state: np.ndarray, 
        next_state: np.ndarray, 
        wall_collision: bool, 
        self_collision: bool, 
        ate_food: bool,
        context: TrainingContext,
        game_instance=None
    ) -> Tuple[float, TrainingStrategy]:
        """Calculate reward using appropriate strategy"""
        strategy = self.get_active_strategy(context)
        reward = strategy.calculate_reward(
            state, next_state, wall_collision, self_collision, ate_food, context, game_instance
        )
        return reward, strategy
        
    def add_strategy(self, strategy: TrainingStrategy):
        """Add a custom training strategy"""
        self.strategies.append(strategy)
        
    def remove_strategy(self, name: str):
        """Remove a strategy by name"""
        self.strategies = [s for s in self.strategies if s.name != name]
        
    def list_strategies(self) -> list:
        """List all available strategies"""
        return [(s.name, s.description) for s in self.strategies]
