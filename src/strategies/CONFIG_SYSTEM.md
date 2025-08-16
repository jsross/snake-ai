# Strategy Configuration System Documentation

## Overview
Each training strategy now has its own configuration file with strategy-specific reward weights and parameters. This allows fine-tuning of individual strategies without affecting others.

## Configuration Files Structure

### Survival Strategy (`survival_config.yaml`)
- **Focus**: Basic survival skills
- **Wall penalty**: -100.0 (doubled from base)
- **Self-collision penalty**: -120.0 (doubled from base)
- **Food reward**: 50.0 (reduced to 50% of base)
- **Exploration**: +20% higher
- **Distance guidance**: Disabled (0.0)

### Food Seeking Strategy (`food_seeking_config.yaml`)
- **Focus**: Learning to move toward food
- **Wall penalty**: -50.0 (standard)
- **Self-collision penalty**: -60.0 (standard)
- **Food reward**: 80.0 (moderate)
- **Exploration**: +10% higher
- **Distance guidance**: Active (1.0 closer, -0.2 farther)

### Advanced Strategy (`advanced_config.yaml`)
- **Focus**: High scores and efficient play
- **Wall penalty**: -50.0 (standard)
- **Self-collision penalty**: -60.0 (standard)
- **Food reward**: 150.0 (1.5x enhanced)
- **Exploration**: -20% lower (more exploitation)
- **Distance guidance**: Enhanced (0.75 closer, -0.12 farther)

### Exploration Strategy (`exploration_config.yaml`)
- **Focus**: Diverse movement patterns
- **Food reward**: 120.0 (1.2x with exploration bonus)
- **Exploration**: +50% higher
- **Distance guidance**: Reduced for exploration
- **Special**: Edge exploration bonus (+0.1)

### Efficiency Strategy (`efficiency_config.yaml`)
- **Focus**: Quick and efficient food collection
- **Food reward**: 200.0 (2x for efficiency)
- **Exploration**: -50% lower (maximum exploitation)
- **Distance guidance**: Strong (1.0 closer, -0.2 farther)
- **Episodes**: 20% shorter for efficiency pressure

## Benefits of Strategy-Specific Configs

1. **Precise Tuning**: Each strategy optimized for its specific learning goal
2. **Easy Adjustment**: Modify individual strategy behavior without affecting others
3. **Clear Separation**: Strategy logic and parameters kept together
4. **Maintainability**: Easy to understand and modify individual strategies
5. **Experimentation**: Test different parameter combinations per strategy

## Usage

Each strategy automatically loads its config file on initialization:
```python
# Strategy loads its own config
strategy = SurvivalTrainingStrategy()
weights = strategy.get_reward_weights()  # Strategy-specific weights
params = strategy.get_strategy_params()  # Strategy-specific parameters
```

No changes needed to existing training code - strategies handle their own configuration internally.
