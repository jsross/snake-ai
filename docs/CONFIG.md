# Configuration Management

The Snake AI system now uses a centralized configuration file to manage all default settings. This replaces the previously hardcoded constants and provides a single place to modify all training parameters.

## Configuration File: `config.yaml`

The configuration is stored in the root directory as `config.yaml` and contains four main sections:

### 1. Reward Weights (`reward_weights`)
Controls the reinforcement learning reward system:
- `wall_collision_penalty`: Penalty for hitting walls (-5.0)
- `self_collision_penalty`: Penalty for snake hitting itself (-6.0)  
- `food_reward`: Reward for eating food (100.0)
- `closer_reward`: Reward for moving toward food (0.5)
- `farther_penalty`: Penalty for moving away from food (-0.1)
- `survival_reward`: Reward for each step survived (0.02)
- `move_cost`: Small cost per move to encourage efficiency (-0.002)
- `no_score_penalty`: Penalty for ending with no score (-10.0)

### 2. Game Configuration (`game_config`)
Controls game mechanics:
- `base_max_steps`: Base maximum steps per episode (500)
- `extra_steps_per_food`: Additional steps granted per food eaten (200)

### 3. Training Configuration (`training_config`)
Controls the training process:
- `epsilon_start`: Starting exploration rate (1.0)
- `epsilon_end`: Ending exploration rate (0.01)
- `epsilon_decay_steps`: Steps over which to decay epsilon (5000)
- `checkpoint_frequency`: Episodes between model saves (500)

### 4. UI Configuration (`ui_config`)
Controls user interface behavior:
- `enable_live_plotting`: Whether to show real-time training plots (false)

## How It Works

1. **Loading**: Both `src/__main__.py` and `src/project_manager.py` load the config on startup using PyYAML
2. **Fallback**: If `config.yaml` is missing or invalid, built-in defaults are used
3. **Project Creation**: New projects use config defaults for their reward weights
4. **Runtime**: Existing projects override defaults with their saved training_config

## Benefits

- **Human-Readable**: YAML format is easier to read and edit than JSON
- **Comments**: YAML supports inline comments for documentation
- **Centralized Management**: All defaults in one place
- **Easy Tuning**: Modify config.yaml without touching code
- **Version Control**: Configuration changes are tracked
- **Project Isolation**: Each project can override defaults
- **Backward Compatibility**: Existing projects continue to work

## Making Changes

To modify default settings:
1. Edit `config.yaml` 
2. Restart the application
3. New projects will use the updated defaults
4. Existing projects keep their current settings

## Error Handling

- Missing config file: Uses hardcoded defaults
- Invalid YAML: Uses hardcoded defaults  
- Missing sections: Uses hardcoded defaults for missing parts
- Error messages are logged for debugging

## Example Configuration

```yaml
# Snake AI Configuration
reward_weights:
  wall_collision_penalty: -5.0      # Penalty for hitting walls
  self_collision_penalty: -6.0      # Penalty for snake hitting itself
  food_reward: 100.0                # Reward for eating food
  closer_reward: 0.5                # Reward for moving toward food
  farther_penalty: -0.1             # Penalty for moving away from food
  survival_reward: 0.02             # Reward for each step survived
  move_cost: -0.002                 # Small cost per move to encourage efficiency
  no_score_penalty: -10.0           # Penalty for ending with no score

game_config:
  base_max_steps: 500               # Base maximum steps per episode
  extra_steps_per_food: 200         # Additional steps granted per food eaten

training_config:
  epsilon_start: 1.0                # Starting exploration rate (100% random)
  epsilon_end: 0.01                 # Ending exploration rate (1% random)
  epsilon_decay_steps: 5000         # Steps over which to decay epsilon
  checkpoint_frequency: 500         # Episodes between model saves

ui_config:
  enable_live_plotting: false       # Whether to show real-time training plots
```
