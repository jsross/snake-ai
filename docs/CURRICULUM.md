# Curriculum Learning System

The Snake AI now implements a **curriculum learning** approach that teaches the AI progressively harder concepts in stages, mimicking how humans learn complex skills.

## Learning Stages

### 🛡️ Stage 1: Survival (Episodes 0-2000 or until 80% survival rate)
**Goal**: Learn basic movement and collision avoidance

**Reward Modifications**:
- **Wall collision penalty**: 2x stronger (-100 instead of -50)
- **Self collision penalty**: 2x stronger (-120 instead of -60)  
- **Food reward**: 0.5x weaker (50 instead of 100)
- **Movement guidance**: Disabled (no closer/farther rewards)
- **Survival bonus**: Extra survival reward added

**Focus**: The AI learns to move around without dying. Food collection is de-emphasized so the AI doesn't get distracted from basic survival skills.

### 🍎 Stage 2: Food Seeking (After survival mastery, until average score > 3.0)
**Goal**: Learn to actively seek and collect food

**Reward Modifications**:
- **Food reward**: Normal strength (100)
- **Movement guidance**: Enabled (closer/farther rewards active)
- **Collision penalties**: Normal strength
- **Focus**: With survival skills mastered, the AI can now learn to navigate toward food

### 🏆 Stage 3: Advanced Play (After consistent food collection)
**Goal**: Optimize strategy and achieve high scores

**Reward Modifications**:
- **Food reward**: 1.5x stronger (150)
- **Movement guidance**: 1.5x stronger
- **All skills combined**: Survival + food seeking + optimization

## Stage Progression Criteria

The system automatically determines the current stage based on:

1. **Episode Count**: Minimum time in survival stage
2. **Survival Rate**: Percentage of recent episodes where the AI got at least 1 food
3. **Average Score**: Recent 100-episode average score

```python
def get_learning_stage(episode, avg_score, survival_rate):
    if episode < 2000 or survival_rate < 0.8:
        return "survival"
    elif avg_score < 3.0:
        return "food_seeking"  
    else:
        return "advanced"
```

## Benefits of This Approach

### **Faster Learning**
- AI masters one skill before moving to the next
- No distraction from competing objectives
- More stable learning progression

### **Better Final Performance**  
- Solid foundation of basic skills
- Less likely to develop bad habits
- More robust behavior patterns

### **Clearer Training Progress**
- Easy to see which stage the AI is in
- Stage transitions mark clear milestones
- Better debugging and analysis

## Training Output

The training log now shows the current stage:

```
Episode: 245, Stage: survival 🛡️, Epsilon: 0.951, Score: 0, Steps: 15, Total Reward: -17.35, Duration: 0.02s
Episode: 1847, Stage: food_seeking 🍎, Epsilon: 0.631, Score: 2, Steps: 45, Total Reward: 185.23, Duration: 0.05s  
Episode: 3204, Stage: advanced 🏆, Epsilon: 0.122, Score: 8, Steps: 120, Total Reward: 1205.67, Duration: 0.12s
```

## Configuration

The curriculum learning system uses the base rewards from `config.yaml` and applies stage-specific multipliers. You can tune the base values to affect all stages proportionally.

**Key Configuration Tips**:
- Higher collision penalties help survival stage
- Moderate food rewards prevent distraction in early stages
- Lower epsilon decay allows more exploration during skill acquisition

## Data Analysis

Training logs now include:
- `learning_stage`: Current curriculum stage
- `survival_rate`: Recent survival performance
- Stage transitions for progress tracking

This enables detailed analysis of how the AI progresses through each learning phase.

## Future Enhancements

Potential improvements to the curriculum system:
- **Adaptive thresholds**: Adjust stage transitions based on learning speed
- **Skill regression detection**: Return to earlier stages if performance drops
- **Custom curriculum**: User-defined learning progressions
- **Multi-skill stages**: Parallel learning of complementary skills
