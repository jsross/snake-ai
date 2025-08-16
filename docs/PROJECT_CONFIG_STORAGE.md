# Project Weight and Configuration Storage

## Overview
The Snake AI project now comprehensively stores all training configurations and weights, ensuring complete reproducibility and tracking.

## What Gets Stored in Projects

### 📁 Project Structure
```
my_snake_project/
├── models/
│   ├── best_model.pth          # Best performing model weights
│   └── checkpoint.pth          # Latest training checkpoint
├── logs/
│   ├── training_data_*.csv     # Episode-by-episode training data
│   └── strategy_analytics_*.json  # Strategy usage analytics
├── plots/
│   └── training_progress.png   # Training progress visualizations
├── config/
│   └── strategies/             # Strategy-specific configurations
│       ├── survival_config.yaml
│       ├── food_seeking_config.yaml
│       ├── advanced_config.yaml
│       ├── exploration_config.yaml
│       ├── efficiency_config.yaml
│       └── strategy_states.json    # Runtime strategy states
└── project.json               # Project metadata and global config
```

### ⚙️ Global Training Configuration
Stored in `project.json` under `training_config`:
- **Epsilon parameters**: Start/end values for exploration
- **Checkpoint frequency**: How often to save progress
- **Global reward weights**: Base reward weight values
- **Model architecture**: Input/output sizes, hidden layers

### 🎯 Strategy-Specific Configurations
Each strategy has its own config file with:

#### **Reward Weights** (specific to each strategy)
- `wall_collision_penalty`: Death penalty for hitting walls
- `self_collision_penalty`: Death penalty for self-collision  
- `food_reward`: Reward for eating food
- `closer_reward`: Reward for moving toward food
- `farther_penalty`: Penalty for moving away from food
- `survival_reward`: Reward for staying alive
- `move_cost`: Cost per move
- `no_score_penalty`: Penalty for ending with no score

#### **Strategy Parameters**
- `exploration_multiplier`: How much to modify exploration rate
- `max_steps_multiplier`: Episode length modification
- `activation_threshold`: When to activate this strategy

### 📊 Training Analytics
Stored in logs directory:
- **Training Data CSV**: Episode scores, rewards, steps, strategy used
- **Strategy Analytics JSON**: Transition analysis, performance metrics
- **Strategy Timeline**: When each strategy was active

### 🔄 Automatic Storage Events
Configurations are saved:
1. **Project Creation**: Initial strategy configs copied to project
2. **Training Start**: Current strategy states saved
3. **Training Completion**: Final strategy states and analytics saved

## Benefits

### ✅ **Complete Reproducibility**
- All training parameters preserved with the model
- Strategy configs can be restored exactly
- Training progression fully documented

### ✅ **Version Control**
- Each project maintains its own config versions
- Strategy modifications tracked over time
- Easy to compare different configuration approaches

### ✅ **Easy Sharing**
- Projects are self-contained with all configs
- Export/import includes all strategy configurations
- No external dependencies on global config files

### ✅ **Analysis & Optimization**
- Strategy performance tracked per configuration
- Easy to identify which weight combinations work best
- Timeline shows strategy progression during training

## Usage Examples

### Inspect Project Configurations
```bash
python inspect_project.py ./projects/my_snake_project
```

### Load Project Strategy Configs
```python
from project_manager import SnakeAIProject

project = SnakeAIProject("./projects/my_snake_project")
project.load_project()

# Get strategy configurations
strategy_configs = project.load_strategy_configs()
survival_config = strategy_configs.get('survival', {})
weights = survival_config.get('reward_weights', {})
```

### Export Project with All Configs
```python
# Creates ZIP with models, configs, logs, and analytics
archive_path = project.export_project_archive()
```

The project now provides complete configuration management, ensuring that every aspect of training is preserved and can be reproduced or analyzed later! 🎯
