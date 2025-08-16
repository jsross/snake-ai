# Snake AI Training Framework

The Snake AI Training Framework provides a pluggable architecture for implementing different training strategies. This allows you to easily experiment with different approaches to curriculum learning and specialized training phases.

## Framework Architecture

### Core Components

#### `TrainingStrategy` (Abstract Base Class)
All training strategies inherit from this base class and must implement:
- `should_activate()` - Determines when this strategy should be used
- `calculate_reward()` - Computes rewards for actions in this strategy
- `get_stage_name()` - Returns human-readable name for the strategy
- `get_stage_emoji()` - Returns emoji representation for the strategy

Optional methods you can override:
- `modify_epsilon()` - Adjust exploration rate for this strategy
- `get_max_steps()` - Modify episode length for this strategy

#### `TrainingFramework` 
Manages all training strategies and automatically selects the appropriate one based on current training context.

#### `TrainingContext` & `TrainingStats`
Provide rich context information to strategies including episode number, performance statistics, and configuration.

## Built-in Strategies

### 1. Survival Training Strategy 🛡️
**Activation**: Episodes 0-2000 OR survival rate < 80%
**Focus**: Learning to avoid walls and self-collision
**Modifications**:
- Doubles death penalties (-50 → -100, -60 → -120)
- Reduces food reward emphasis (×0.5)
- Increases exploration rate (+20%)
- Adds survival bonus

### 2. Food Seeking Training Strategy 🍎  
**Activation**: After survival mastery, before advanced play (avg_score < 3.0)
**Focus**: Learning to move toward food while maintaining survival
**Modifications**:
- Normal death penalties and food rewards
- Adds distance-based guidance (closer/farther rewards)
- Standard exploration rate

### 3. Advanced Training Strategy 🎯
**Activation**: High performance (survival_rate ≥ 80%, avg_score ≥ 3.0)  
**Focus**: Optimizing for high scores and efficient play
**Modifications**:
- Enhanced food rewards (×1.5)
- Stronger distance-based guidance (×1.5)
- Reduced exploration rate (-20%)

## Creating Custom Strategies

### Basic Custom Strategy

```python
from training_framework import TrainingStrategy, TrainingContext
import numpy as np

class MyCustomStrategy(TrainingStrategy):
    def __init__(self):
        super().__init__(
            name="my_strategy",
            description="My custom training approach"
        )
        
    def should_activate(self, context: TrainingContext) -> bool:
        # Define when this strategy should be active
        return context.episode >= 1000 and context.stats.avg_score >= 2.0
        
    def get_stage_name(self) -> str:
        return "My Custom Stage"
        
    def get_stage_emoji(self) -> str:
        return "⭐"
        
    def calculate_reward(self, state, next_state, wall_collision, 
                        self_collision, ate_food, context):
        weights = context.reward_weights
        
        # Implement your custom reward logic here
        if wall_collision or self_collision:
            return weights['wall_collision_penalty']
            
        if ate_food:
            return weights['food_reward'] * 1.2  # Custom multiplier
            
        return weights['survival_reward'] + weights['move_cost']
```

### Adding Strategies to the Framework

```python
# In __main__.py or a separate module
from custom_strategies import MyCustomStrategy

# Add to the global framework
training_framework.add_strategy(MyCustomStrategy())

# Or add multiple strategies
from custom_strategies import add_custom_strategies
add_custom_strategies(training_framework)
```

## Advanced Features

### Strategy Priority
Strategies are evaluated in reverse order (most advanced first). The first strategy that returns `True` from `should_activate()` is used.

### Context-Aware Training
Each strategy receives rich context including:
- Current episode number
- Recent performance statistics (avg_score, survival_rate, avg_reward)
- Current reward weights
- Full configuration

### Epsilon Modification
Strategies can adjust exploration rate:

```python
def modify_epsilon(self, base_epsilon: float, context: TrainingContext) -> float:
    if context.stats.avg_score < 1.0:
        return base_epsilon * 1.5  # More exploration
    else:
        return base_epsilon * 0.7  # Less exploration
```

### Episode Length Control
Strategies can modify episode length:

```python
def get_max_steps(self, base_steps: int, extra_per_food: int, score: int) -> int:
    # Shorter episodes for efficiency training
    return int((base_steps + extra_per_food * score) * 0.8)
```

## Example Custom Strategies

See `custom_strategies.py` for examples:

### Exploration Strategy 🔍
- **Purpose**: Encourage diverse movement patterns
- **Activation**: Episodes 1000-3000
- **Features**: Edge exploration bonus, increased epsilon

### Efficiency Strategy ⚡
- **Purpose**: Optimize food collection speed
- **Activation**: High performance (episodes ≥ 5000, avg_score ≥ 5.0)
- **Features**: Double food rewards, strong guidance, shorter episodes

## Benefits of the Framework

1. **Modularity**: Each strategy is self-contained and easy to understand
2. **Extensibility**: Add new strategies without modifying core code
3. **Testability**: Strategies can be tested in isolation
4. **Flexibility**: Mix and match strategies, adjust activation conditions
5. **Reusability**: Strategies can be shared and reused across projects

## Integration with Project System

The framework automatically integrates with the existing project system:
- Strategy information is logged in training data
- Current strategy is displayed during training
- All framework features work with project-based organization

## Future Extensions

The framework is designed to support future enhancements:
- Multi-strategy blending (weighted combinations)
- Dynamic strategy switching within episodes
- Performance-based strategy selection
- Hyperparameter optimization per strategy
- Strategy-specific model architectures
