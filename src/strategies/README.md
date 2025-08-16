# Adding Custom Training Strategies

This framework uses a modular strategy pattern with individual config files for each strategy.

## Quick Start

1. Create a new config file in `strategies/configs/` (e.g., `my_strategy_config.yaml`)
2. Create a new strategy file in `strategies/` (e.g., `my_strategy.py`)
3. Import the base class: `from . import TrainingStrategy, TrainingContext`
4. Implement the required methods
5. Import your strategy in `training_framework.py` and add it to the list

## Example Strategy Config Template

Create `strategies/configs/my_strategy_config.yaml`:
```yaml
# My Custom Strategy Configuration
reward_weights:
  wall_collision_penalty: -50.0
  self_collision_penalty: -60.0
  food_reward: 100.0
  closer_reward: 0.5
  farther_penalty: -0.1
  survival_reward: 0.1
  move_cost: -0.01
  no_score_penalty: -20.0

strategy_params:
  exploration_multiplier: 1.0
  max_steps_multiplier: 1.0
  activation_threshold:
    min_episode: 1000
    min_survival_rate: 0.7
```

## Example Strategy Template

```python
from . import TrainingStrategy, TrainingContext

class MyCustomStrategy(TrainingStrategy):
    def __init__(self):
        super().__init__(
            name="my_strategy",
            description="Description of what this strategy does",
            config_name="my_strategy_config.yaml"
        )
    
    def should_activate(self, context: TrainingContext) -> bool:
        """Define when this strategy should be active"""
        params = self.get_strategy_params()
        threshold = params.get('activation_threshold', {})
        
        return (context.episode >= threshold.get('min_episode', 1000) and 
                context.stats.survival_rate >= threshold.get('min_survival_rate', 0.7))
    
    def calculate_reward(self, state, next_state, wall_collision, 
                        self_collision, ate_food, context) -> float:
        """Define custom reward calculation using strategy config"""
        weights = self.get_reward_weights()
        
        if wall_collision:
            return weights.get('wall_collision_penalty', -50.0)
        if self_collision:
            return weights.get('self_collision_penalty', -60.0)
        if ate_food:
            return weights.get('food_reward', 100.0)
            
        return weights.get('survival_reward', 0.1) + weights.get('move_cost', -0.01)
    
    def get_stage_name(self) -> str:
        return "My Custom Training"
```

## Available Context Data

The `TrainingContext` provides:
- `episode`: Current episode number
- `step`: Current step in episode
- `score`: Current score
- `snake_length`: Current snake length
- `food_eaten`: Whether food was eaten this step
- `done`: Whether episode ended
- `reason`: Why episode ended ("wall", "self", None)
- `survival_rate`: Recent survival rate
- `avg_score`: Recent average score
- `avg_length`: Recent average length

## Integration

To use your strategy, add it to `training_framework.py`:

```python
from strategies.my_strategy import MyCustomStrategy

class TrainingFramework:
    def __init__(self):
        self.strategies = [
            SurvivalTrainingStrategy(),
            FoodSeekingTrainingStrategy(), 
            AdvancedTrainingStrategy(),
            MyCustomStrategy(),  # Add your strategy here
        ]
```

## Tips

- Start with survival-focused strategies for early training
- Use higher exploration rates for learning phases
- Gradually increase complexity as training progresses
- Test your strategy logic with different context values
- Monitor training metrics to validate strategy effectiveness
