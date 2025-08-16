# Strategy Selection Guide

## Overview

The Snake AI framework now supports **manual strategy selection** for training! You can choose which training strategy to use instead of relying solely on automatic curriculum learning.

## Available Strategies

### 1. Auto (Curriculum Learning) 
- **Default behavior** - automatically selects the best strategy based on current performance
- Starts with Survival → progresses to Food Seeking → advances to Advanced strategy
- Recommended for most training sessions

### 2. Survival Strategy
- **Focus**: Pure survival - avoiding walls and self-collision
- **Best for**: Initial training, learning basic movement
- **Rewards**: 
  - Wall collision: -100
  - Self collision: -120  
  - Survival step: +0.2
  - No food rewards (pure survival focus)

### 3. Food Seeking Strategy
- **Focus**: Learning to move toward food while avoiding death
- **Best for**: Intermediate training, learning food acquisition
- **Rewards**: Balanced approach with food rewards and distance guidance

### 4. Advanced Strategy  
- **Focus**: Optimizing for high scores and efficient play
- **Best for**: Advanced training, maximizing performance
- **Rewards**: Higher food rewards and sophisticated scoring

## How to Use Strategy Selection

### In the Application

1. **Start the application**:
   ```bash
   python src/__main__.py
   ```

2. **Load or create a project** (required for training)

3. **Select "Training Mode"** from the main menu

4. **Choose your strategy** from the strategy selection menu:
   - Use ↑/↓ arrow keys to navigate
   - Press Enter to select
   - Press Esc to go back
   - Or press number keys (1-4) for quick selection

5. **Enter training iterations** and start training

### Strategy Selection Menu

```
Select Training Strategy

1. Auto (Curriculum Learning)
2. Survival Strategy  
3. Food Seeking Strategy
4. Advanced Strategy

Use UP/DOWN arrows to navigate
Press ENTER to select
Press ESC to go back
```

## When to Use Each Strategy

### Use **Auto** when:
- ✅ You want the system to automatically progress through difficulty levels
- ✅ Training a new model from scratch
- ✅ You're unsure which strategy to use

### Use **Survival** when:
- ✅ Model is struggling with basic movement
- ✅ High collision rate needs to be addressed
- ✅ Want to focus purely on staying alive

### Use **Food Seeking** when:
- ✅ Model can survive but doesn't seek food effectively
- ✅ Want to improve food acquisition without complexity
- ✅ Balancing survival and growth

### Use **Advanced** when:
- ✅ Model has mastered basics and survival
- ✅ Want to maximize score and efficiency
- ✅ Fine-tuning for optimal performance

## Strategy Analytics

All strategy usage is automatically tracked in the analytics system:

- **Strategy transitions** and timing
- **Performance per strategy** 
- **Episode tracking** with strategy context
- **Analytics export** for analysis

View analytics with:
```python
from strategy_analytics import strategy_tracker
strategy_tracker.export_analytics()
```

## Testing Strategy Selection

Run the test script to verify all strategies work correctly:

```bash
python test_strategy_selection.py
```

This will test:
- Strategy selection functionality
- Reward calculation for each strategy
- Automatic vs. manual strategy modes

## Tips for Effective Training

1. **Start with Auto mode** for new models to establish baseline performance

2. **Use Survival strategy** if you notice high collision rates in analytics

3. **Switch to Food Seeking** when survival rate > 80% but food acquisition is low

4. **Use Advanced strategy** for final optimization when model performs well

5. **Monitor analytics** to understand which strategies work best for your specific model

## Configuration

Each strategy has its own configuration file in `src/strategies/configs/`:
- `survival_config.yaml` - Survival strategy settings
- `food_seeking_config.yaml` - Food seeking parameters  
- `advanced_config.yaml` - Advanced strategy configuration

These configs are automatically loaded and stored with your projects.
