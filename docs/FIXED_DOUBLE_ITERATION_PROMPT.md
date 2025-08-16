# Fixed: Double Iteration Prompt Issue

## Problem
The training mode was asking for iteration count **twice**:
1. First in `choose_mode_pygame()` when TRAINING_MODE was selected
2. Second in the enhanced training flow after strategy/weight selection

## Solution
Modified `menu.py` line 92 to **not** ask for iterations when TRAINING_MODE is selected:

**Before:**
```python
if selected_key == TRAINING_MODE:
    iterations = get_training_iterations(display, font, game)  # Asked here
```

**After:**
```python
if selected_key == TRAINING_MODE:
    # Don't ask for iterations here - handled in main flow with strategy selection
    iterations = None  # Skip asking here
```

## New Training Flow
Now the training flow works correctly:

1. **Select "Training Mode"** → No iteration prompt
2. **Choose Strategy** → Auto/Survival/Food Seeking/Advanced  
3. **Configure Weights** → Edit weights (if not auto mode)
4. **Set Iterations** → **Single prompt** for training iterations
5. **Start Training** → Progress bar and training begins

## Result
✅ **Single iteration prompt** - only asked once after strategy selection
✅ **Clean flow** - strategy → weights → iterations → training
✅ **No duplication** - fixed the double prompting issue

The enhanced training system now works smoothly without asking for iterations twice!
