# Move Cost Reward Implementation

## Overview
The Snake AI now includes a **move cost** in its reward calculation to encourage more efficient exploration and faster convergence to optimal solutions.

## What Changed

### 1. Project Configuration
- Added `move_cost` parameter to the default project configuration in `project_manager.py`
- Default value: `-0.01` (small cost per move)

### 2. Reward Calculation
- Modified `calculate_reward()` function in `__main__.py` to include move cost
- Each move now costs a small amount of reward points
- This encourages the snake to find shorter paths to food

### 3. Dynamic Configuration
- Reward weights are now loaded from project configuration
- If no project is loaded, defaults to hardcoded values
- Move cost is configurable per project

## How Move Cost Works

### Previous Behavior
```python
# Old reward per normal step
reward = survival_reward + step_penalty
# Example: 0.01 + (-0.01) = 0.0
```

### New Behavior
```python
# New reward per normal step
reward = survival_reward + move_cost
# Example: 0.01 + (-0.01) = 0.0 (same net effect but semantically different)
```

### Impact on Learning
1. **Efficiency**: Snake learns to take shorter paths to food
2. **Exploration**: Discourages aimless wandering
3. **Convergence**: Should lead to faster learning of optimal strategies

## Configuration Examples

### Low Move Cost (More Exploration)
```json
"reward_weights": {
    "move_cost": -0.005
}
```

### Higher Move Cost (More Aggressive Efficiency)
```json
"reward_weights": {
    "move_cost": -0.02
}
```

### No Move Cost (Original Behavior)
```json
"reward_weights": {
    "move_cost": 0.0
}
```

## Testing the Implementation

Run the test script to verify the implementation:
```bash
python test_reward_weights.py
```

This will show:
- Default reward weights including move cost
- Project configuration with move cost
- Reward calculation examples

## Expected Benefits

1. **Faster Training**: Snake learns optimal paths quicker
2. **Better Performance**: Reduced average steps to reach food
3. **More Realistic Behavior**: In real scenarios, movement has energy cost
4. **Configurable Balance**: Can tune the trade-off between exploration and efficiency

## Usage

The move cost is automatically applied when training the AI. To adjust it:

1. Create or load a project
2. Modify the `reward_weights.move_cost` value in the project configuration
3. More negative values = higher cost = more efficiency pressure
4. Values closer to 0 = lower cost = more exploration allowed

## Implementation Details

- Move cost is applied on every step (except death or food consumption)
- Combined with survival reward to determine net step reward
- Configurable through project system for easy experimentation
- Backward compatible with existing projects (defaults applied)
