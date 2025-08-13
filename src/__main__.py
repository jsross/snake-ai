import math
import os
import sys
import time
import torch
import pygame
import numpy as np
from pygame.locals import *

# Handle both relative and absolute imports
try:
    from .snake_game import SnakeGame, Action
    from .snake_ai import SnakeAI
    from .snake_game_renderer import render
    from .menu import choose_mode_pygame, CREATE_NEW_MODEL, AI_DEMO, MANUAL_MODE, TRAINING_MODE, LOAD_MODEL_INTERACTIVE, RESUME
    from .utils import load_checkpoint, save_unified_model, load_unified_model
    from .model_container import SnakeModelContainer
    from .project_manager import SnakeAIProject, ProjectManager, interactive_project_selection
except ImportError:
    # Fallback to absolute imports when run directly
    from snake_game import SnakeGame, Action
    from snake_ai import SnakeAI
    from snake_game_renderer import render
    from menu import choose_mode_pygame, CREATE_NEW_MODEL, AI_DEMO, MANUAL_MODE, TRAINING_MODE, LOAD_MODEL_INTERACTIVE, RESUME
    from utils import load_checkpoint, save_unified_model, load_unified_model
    from model_container import SnakeModelContainer
    from project_manager import SnakeAIProject, ProjectManager, interactive_project_selection

pygame.init()
game = SnakeGame(initial_snake_length=8)  # Start with a longer snake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

display = pygame.display.set_mode((game.w, game.h))
display.fill((0, 0, 0))
pygame.display.set_caption('Snake AI')
font = pygame.font.SysFont('arial', 25)
clock = pygame.time.Clock()

input_size = game.get_cell_count()
output_size = 3

ai_model = SnakeAI(input_size=input_size, output_size=output_size)
ai_model.model.to(device)
ai_model.target_model.to(device)
resume_requested = False

episode_scores = []
episode_rewards = []
average_scores = []
average_rewards = []

# CSV logging for training analysis
training_log = []  # Store training data for CSV export

# Plot control
ENABLE_LIVE_PLOTTING = False  # Set to True to enable live plotting during training (may cause crashes)

# Checkpointing variables
best_reward = float('-inf')  # Use reward instead of score for better model selection
best_score = 0  # Keep for display purposes
episode = 0  # Initialize episode counter

# Model status tracking - now uses project system
current_project = None
current_model_info = {
    'loaded': False,
    'project_name': None,
    'episode': 0
}

# Initialize paths (will be set when project is loaded)
best_model_path = None
checkpoint_path = None
models_dir = None
checkpoint_frequency = 500  # Save checkpoint every N episodes

resume_requested = False

# Reward weights - More stable reward structure
WALL_COLLISION_PENALTY = -50.0  # Strong penalty for dying
SELF_COLLISION_PENALTY = -50.0  # Strong penalty for self-collision
FOOD_REWARD = 100.0  # High reward for food
CLOSER_REWARD = 0.05  # Very small reward for moving toward food
FARTHER_PENALTY = -0.05  # Very small penalty for moving away
SURVIVAL_REWARD = 0.01  # Small survival reward
STEP_PENALTY = -0.01  # Very small step penalty

# Episode step control
BASE_MAX_STEPS = 500  # Much longer episodes for more learning
EXTRA_STEPS_PER_FOOD = 200

def calculate_reward(state, next_state, wall_collision, self_collision, ate_food):
    """Calculate reward with improved structure to prevent spinning"""
    
    # Death penalties - strong negative signal
    if wall_collision or self_collision:
        return WALL_COLLISION_PENALTY
    
    # Food reward - strong positive signal
    if ate_food:
        return FOOD_REWARD

    # Base survival reward
    reward = SURVIVAL_REWARD + STEP_PENALTY
    
    # Distance-based guidance (much smaller to prevent spinning)
    head_pos = np.argwhere(state == 1)[0]
    next_head_pos = np.argwhere(next_state == 1)[0]
    food_pos = np.argwhere(state < 0)[0]
    prev_distance = np.linalg.norm(head_pos - food_pos)
    new_distance = np.linalg.norm(next_head_pos - food_pos)

    if new_distance < prev_distance:
        reward += CLOSER_REWARD
    elif new_distance > prev_distance:
        reward += FARTHER_PENALTY

    return reward

def run_episode(mode, epsilon, train=False):
    state = game.reset()
    done = False
    steps = 0
    total_reward = 0.0
    debug_actions = []  # Track actions for debugging
    recent_actions = []  # Track recent actions to prevent spinning

    max_steps = BASE_MAX_STEPS

    while not done and steps < max_steps:
        if mode == "demo" or train:
            action_idx = ai_model.get_action(state.flatten(), epsilon, device)
            action = Action(action_idx)
            debug_actions.append(action_idx)
            recent_actions.append(action_idx)
            
            # Keep only last 10 actions for spinning detection
            if len(recent_actions) > 10:
                recent_actions.pop(0)
        else:
            action = Action.STRAIGHT
            action_idx = 0

        wall_collision, self_collision, ate_food = game.step(action)
        next_state = game.get_state()

        done = wall_collision or self_collision
        
        max_steps = BASE_MAX_STEPS + game.score * EXTRA_STEPS_PER_FOOD

        reward = calculate_reward(state, next_state, wall_collision, self_collision, ate_food)
        
        # Anti-spinning penalty: penalize if too many turns in recent actions
        if mode == "training" and len(recent_actions) >= 6:
            # Count non-straight actions in recent moves
            turn_count = sum(1 for a in recent_actions[-6:] if a != 0)
            if turn_count >= 5:  # If 5 out of last 6 moves were turns
                reward -= 2.0  # Spinning penalty

        total_reward += reward

        if mode == "training":
            ai_model.remember(state.flatten(), action_idx, reward, next_state.flatten(), done)

            # Train less frequently for more stable learning
            if len(ai_model.memory) > 128 and steps % 4 == 0:  # Less frequent, larger memory buffer
                ai_model.train(device)
        else:
            render(display, next_state, game.score, font)
            clock.tick(40)

        state = next_state
        steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return game.score, steps, total_reward, True

    # Debug output for demo mode
    if mode == "demo" and len(debug_actions) > 0:
        action_counts = {0: debug_actions.count(0), 1: debug_actions.count(1), 2: debug_actions.count(2)}
        print(f"Actions taken - Straight: {action_counts[0]}, Left: {action_counts[1]}, Right: {action_counts[2]}")
        
        # Show first 20 actions to help identify patterns
        if len(debug_actions) >= 20:
            print(f"First 20 actions: {debug_actions[:20]}")
        else:
            print(f"All actions: {debug_actions}")
            
        # Calculate action distribution percentages
        total_actions = len(debug_actions)
        print(f"Action distribution: Straight {action_counts[0]/total_actions*100:.1f}%, Left {action_counts[1]/total_actions*100:.1f}%, Right {action_counts[2]/total_actions*100:.1f}%")

    return game.score, steps, total_reward, False

def create_new_project():
    """Create a new project with user-selected location via file dialog."""
    global current_project, best_model_path, checkpoint_path, models_dir, current_model_info, episode, best_reward, best_score
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Use Documents directory as default location
        projects_dir = os.path.join(os.path.expanduser("~"), "Documents")
        
        root = tk.Tk()
        root.withdraw()
        
        # Use save dialog to let user choose project location and name
        project_file = filedialog.asksaveasfilename(
            title="Create New Snake AI Project",
            defaultextension="",
            filetypes=[
                ("Project Folders", "*"),
                ("All Files", "*.*")
            ],
            initialdir=projects_dir,
            initialfile="Snake_AI_Project"
        )
        
        root.destroy()
        
        if not project_file:
            return False
        
        # Use the selected path as project directory (remove any extension)
        project_path = os.path.splitext(project_file)[0]
        
        # Create the project
        current_project = SnakeAIProject(project_path)
        
        if current_project.create_project():
            # Set up paths
            best_model_path = str(current_project.best_model_path)
            checkpoint_path = str(current_project.checkpoint_path)
            models_dir = str(current_project.project_path)
            
            # Reset training state
            episode = 0
            best_reward = float('-inf')
            best_score = 0
            
            # IMPORTANT: Reset the AI model to start fresh
            print("Resetting AI model for new project...")
            ai_model.model = ai_model.model.__class__(input_size, output_size)  # Create fresh model
            ai_model.target_model = ai_model.target_model.__class__(input_size, output_size)  # Create fresh target model
            ai_model.model.to(device)
            ai_model.target_model.to(device)
            ai_model.optimizer = torch.optim.Adam(ai_model.model.parameters(), lr=ai_model.learning_rate)
            ai_model.memory.clear()  # Clear experience replay memory
            
            # Clear training history
            global episode_scores, episode_rewards, average_scores, average_rewards, training_log
            episode_scores = []
            episode_rewards = []
            average_scores = []
            average_rewards = []
            training_log = []
            
            print("✓ AI model reset to random weights")
            print("✓ Training history cleared")
            
            # Update model info
            current_model_info = {
                'loaded': True,
                'project_name': current_project.project_name,
                'episode': 0
            }
            
            # Store model architecture info
            current_project.metadata["model_architecture"]["input_size"] = input_size
            current_project.metadata["model_architecture"]["output_size"] = output_size
            current_project.save_metadata()
            
            print(f"✓ New project created: {current_project.project_name}")
            print(f"✓ Project location: {current_project.project_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Failed to create new project: {e}")
        return False 

def load_existing_project():
    """Load an existing project with user interaction."""
    global current_project, best_model_path, checkpoint_path, models_dir, current_model_info, episode, best_reward, best_score
    
    try:
        current_project = interactive_project_selection()
        
        if current_project:
            # Set up paths
            best_model_path = str(current_project.best_model_path)
            checkpoint_path = str(current_project.checkpoint_path)
            models_dir = str(current_project.project_path)
            
            # Load the best model for demo/inference
            best_model_loaded = False
            if current_project.best_model_path.exists():
                success, container = load_unified_model(best_model_path, ai_model.model, ai_model.optimizer, device=device)
                if success and container:
                    print(f"✓ Loaded best model from project (Reward: {container.total_reward:.2f})")
                    best_model_loaded = True
                else:
                    print("No valid model found in project - will train from scratch")
            
            # Get training progress info from checkpoint
            episode = current_project.metadata.get("total_episodes", 0)
            best_reward = current_project.metadata.get("best_reward", float('-inf'))
            best_score = current_project.metadata.get("best_score", 0)
            
            if current_project.checkpoint_path.exists():
                try:
                    # Just read checkpoint info without loading it into main model
                    success, container = load_unified_model(checkpoint_path, ai_model.model, ai_model.optimizer, device=device)
                    if success and container:
                        episode = container.episode
                        best_score = container.score
                        best_reward = container.total_reward if hasattr(container, 'total_reward') else container.best_reward
                        print(f"Checkpoint info - Episode: {episode}, Best Score: {best_score}, Best Reward: {best_reward:.2f}")
                        
                        # Only use checkpoint model if no best model exists
                        if not best_model_loaded:
                            print("Using checkpoint model for training continuation")
                        else:
                            # Reload best model to ensure it's active
                            load_unified_model(best_model_path, ai_model.model, ai_model.optimizer, device=device)
                            print("Reloaded best model for demo use")
                    else:
                        print("No valid checkpoint found")
                except Exception as e:
                    print(f"Checkpoint reading failed: {e}")
                    if not best_model_loaded:
                        print("Starting fresh with project")
            
            # Update model info
            current_model_info = {
                'loaded': True,
                'project_name': current_project.project_name,
                'episode': episode
            }
            
            print(f"✓ Project loaded: {current_project.project_name}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Failed to load project: {e}")
        return False

# === MAIN LOOP ===
epsilon_start = 1.0
epsilon_end = 0.01  # Lower minimum epsilon for more exploitation
epsilon_decay_steps = 5000  # Decay over specific number of steps instead of episodes
last_mode = None

while True:
    if not resume_requested:
        mode, training_iterations = choose_mode_pygame(display, font, game, current_model_info)
        if mode == RESUME:
            mode = last_mode
        elif mode == CREATE_NEW_MODEL:
            if create_new_project():
                print("New project created successfully!")
            else:
                print("Project creation cancelled.")
            continue  # Go back to menu
        elif mode == LOAD_MODEL_INTERACTIVE:
            if load_existing_project():
                print("Project loaded successfully!")
            else:
                print("Project loading cancelled.")
            continue  # Go back to menu
        else:
            last_mode = mode

    resume_requested = False

    # Check if model is required for the selected mode
    if mode in [TRAINING_MODE, AI_DEMO] and not current_model_info['loaded']:
        print(f"Error: {mode} requires a model to be loaded or created first.")
        continue

    if mode == TRAINING_MODE:
        training_start_time = time.time()  # Track total training time
        training_session_episodes = 0  # Track episodes in this session

        for i in range(training_iterations):
            # Better epsilon decay - more gradual and longer exploration
            current_step = episode + i  # Use total step count across all training
            if current_step < epsilon_decay_steps:
                epsilon = epsilon_start - (epsilon_start - epsilon_end) * (current_step / epsilon_decay_steps)
            else:
                epsilon = epsilon_end
                
            start_time = time.time()
            score, steps, total_reward, _ = run_episode(mode, epsilon=epsilon, train=True)
            
            # Single additional training step at end of episode for stability
            if len(ai_model.memory) > 128:
                ai_model.train(device)  # Just one additional training step

            episode += 1
            training_session_episodes += 1
            episode_scores.append(score)
            episode_rewards.append(total_reward)
            average_scores.append(np.mean(episode_scores[-100:]))
            average_rewards.append(np.mean(episode_rewards[-100:]))
            
            # Log training data for analysis
            training_log.append({
                'episode': episode,
                'score': score,
                'steps': steps,
                'total_reward': total_reward,
                'epsilon': epsilon,
                'duration': time.time() - start_time,
                'avg_score_100': average_scores[-1],
                'avg_reward_100': average_rewards[-1]
            })

            # Save unified model if reward improved
            if total_reward > best_reward:
                best_reward = total_reward
                best_score = score  # Update best score for display
                
                # Save using unified model system in project
                save_unified_model(
                    ai_model.model, 
                    ai_model.optimizer,
                    best_model_path,
                    episode=episode,
                    score=score,
                    total_reward=total_reward,
                    training_data={
                        'episode_scores': episode_scores[-100:],  # Last 100 scores
                        'episode_rewards': episode_rewards[-100:],  # Last 100 rewards
                        'average_scores': average_scores[-100:] if len(average_scores) >= 100 else average_scores,
                        'average_rewards': average_rewards[-100:] if len(average_rewards) >= 100 else average_rewards,
                        'epsilon': epsilon,
                        'training_duration': time.time() - training_start_time
                    },
                    description=f"Best model - Episode {episode}, Score {score}, Reward {total_reward:.2f}"
                )
                # Update model info and project metadata
                current_model_info['episode'] = episode
                if current_project:
                    current_project.metadata["best_score"] = best_score
                    current_project.metadata["best_reward"] = best_reward
                    current_project.save_metadata()
                print(f"New best reward: {best_reward:.2f} (Score: {score})! Model saved to project.")

            # Save checkpoint periodically using unified system
            if episode % checkpoint_frequency == 0:
                save_unified_model(
                    ai_model.model, 
                    ai_model.optimizer,
                    checkpoint_path,
                    episode=episode,
                    score=score,
                    total_reward=total_reward,
                    training_data={
                        'episode_scores': episode_scores[-100:],  # Last 100 scores
                        'episode_rewards': episode_rewards[-100:],  # Last 100 rewards
                        'average_scores': average_scores[-100:] if len(average_scores) >= 100 else average_scores,
                        'average_rewards': average_rewards[-100:] if len(average_rewards) >= 100 else average_rewards,
                        'epsilon': epsilon,
                        'training_duration': time.time() - training_start_time
                    },
                    description=f"Checkpoint - Episode {episode}, Score {score}, Reward {total_reward:.2f}"
                )
                # Update model info and project metadata
                current_model_info['episode'] = episode
                if current_project:
                    current_project.metadata["total_episodes"] = episode
                    current_project.save_metadata()
                print(f"Checkpoint saved at episode {episode}")

            if i % 100 == 0 and i > 0 and ENABLE_LIVE_PLOTTING:
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.clear()
                    ax.set_title("Training Progress")
                    ax.set_xlabel("Episode")
                    ax.set_ylabel("Average Reward")
                    ax.grid(True)
                    ax.plot(average_rewards, label='Avg Reward (100)', color='green')
                    ax.legend()
                    plt.draw()
                    plt.pause(0.001)  # Very short pause
                except Exception as e:
                    print(f"Plot update failed: {e}")
                    # Continue without crashing

            print(f"Episode: {episode}, Epsilon: {epsilon:.3f}, Score: {score}, Steps: {steps}, Total Reward: {total_reward:.2f}, Duration: {time.time() - start_time:.2f}s")

        # Save final training plot to project
        if current_project:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_title("Training Progress - Final Results")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Average Reward")
                ax.grid(True)
                ax.plot(episode_rewards, label='Episode Rewards', alpha=0.3, color='lightgreen')
                ax.plot(average_rewards, label='Average Reward (100)', color='green', linewidth=2)
                ax.legend()
                plt.savefig(str(current_project.training_plot_path), dpi=150, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory
                print(f"Training plot saved to {current_project.training_plot_path}")
            except Exception as e:
                print(f"Failed to save training plot: {e}")

            # Save training data to CSV in project logs directory
            if training_log:
                try:
                    csv_filename = current_project.get_next_log_filename("training_data")
                    
                    try:
                        import pandas as pd
                        df = pd.DataFrame(training_log)
                        df.to_csv(csv_filename, index=False)
                        print(f"Training data saved to {csv_filename}")
                    except ImportError:
                        print("Pandas not available. Saving training data as basic CSV...")
                        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                            import csv
                            if training_log:
                                fieldnames = training_log[0].keys()
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(training_log)
                                print(f"Training data saved to {csv_filename}")
                    
                    # Add training session to project history
                    session_duration = time.time() - training_start_time
                    current_project.add_training_session(
                        training_session_episodes, 
                        best_score, 
                        best_reward, 
                        session_duration
                    )
                    
                except Exception as e:
                    print(f"Failed to save training data: {e}")
        else:
            print("Warning: No project loaded, training data not saved to project")

    elif mode == AI_DEMO:
        # AI Demo mode - use currently loaded model
        print("AI Demo mode - using currently loaded model")
        if current_project and current_project.best_model_path.exists():
            print(f"Model file: {current_project.best_model_path}")
        
        # Ensure model is in evaluation mode for demo
        ai_model.model.eval()
        
        playing = True
        while playing:
            start_time = time.time()
            score, steps, total_reward, esc_pressed = run_episode(mode="demo", epsilon=0.0)  # No exploration in demo
            episode += 1
            print(f"AI Demo Episode: {episode}, Score: {score}, Steps: {steps}, Total Reward: {total_reward:.2f}, Duration: {time.time() - start_time:.2f}s")

            if esc_pressed:
                resume_requested = False
                playing = False
        
        # Set back to training mode if needed
        ai_model.model.train()

    else:
        # Manual mode
        playing = True
        while playing:
            start_time = time.time()
            score, steps, total_reward, esc_pressed = run_episode(mode=mode, epsilon=epsilon_start)
            episode += 1
            print(f"Episode: {episode}, Score: {score}, Steps: {steps}, Total Reward: {total_reward:.2f}, Duration: {time.time() - start_time:.2f}s")

            if esc_pressed:
                resume_requested = False
                playing = False
