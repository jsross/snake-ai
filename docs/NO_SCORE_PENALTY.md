# No Score Penalty Implementation

## Summary
Added a configurable penalty that is applied when an episode ends with zero score (no food eaten). This encourages the snake to at least attempt to get food rather than just surviving or dying immediately.

## Changes Made

### 1. Project Configuration (`project_manager.py`)
- Added `"no_score_penalty": -10.0` to the default reward weights
- This value is configurable per project

### 2. Reward System (`__main__.py`)
- Added `DEFAULT_NO_SCORE_PENALTY = -20.0` for fallback when no project is loaded
- Updated `get_reward_weights()` to include the no score penalty
- Modified `run_episode()` to apply the penalty at episode end

### 3. Application Logic
The penalty is applied when:
- Episode ends (either by collision or max steps reached)
- Game score is 0 (no food was eaten)
- Mode is "training" (not applied in demo mode)

## How It Works

### Before (without no score penalty):
- Episode with 0 food, hits wall: `step_rewards + wall_penalty = -50.0`
- Episode with 1 food, hits wall: `step_rewards + food_reward + wall_penalty = 50.0`
- Difference: 100.0 points

### After (with no score penalty):
- Episode with 0 food, hits wall: `step_rewards + wall_penalty + no_score_penalty = -70.0`
- Episode with 1 food, hits wall: `step_rewards + food_reward + wall_penalty = 50.0`
- Difference: 120.0 points

## Benefits
1. **Stronger Learning Signal**: 20% more incentive to get food vs dying with no score
2. **Reduced Immediate Deaths**: Discourages strategies that immediately hit walls
3. **Better Exploration**: Encourages the snake to explore for food rather than just survive
4. **Configurable**: Can adjust penalty strength for different training objectives

## Configuration Examples
- `-5.0`: Light penalty, gentle encouragement
- `-10.0`: Default project setting, moderate pressure
- `-20.0`: Default fallback, strong encouragement
- `-50.0`: Very strong penalty, maximum pressure to get food

The penalty value should be balanced against other rewards to maintain effective learning without being too harsh.
