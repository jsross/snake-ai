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

# Reward weights
WALL_COLLISION_PENALTY = -1.0
SELF_COLLISION_PENALTY = -0.5
FOOD_REWARD = 10.0
CLOSER_REWARD = 1.0
FARTHER_PENALTY = -0.5
SURVIVAL_REWARD = -0.01  # Encourage faster food seeking


# Episode step control
BASE_MAX_STEPS = 50
EXTRA_STEPS_PER_FOOD = 50

def calculate_reward(state, next_state, wall_collision, self_collision, ate_food):
    head_pos = np.argwhere(state == 1)[0]
    next_head_pos = np.argwhere(next_state == 1)[0]
    food_pos = np.argwhere(state < 0)[0]
    prev_distance = np.linalg.norm(head_pos - food_pos)
    new_distance = np.linalg.norm(next_head_pos - food_pos)

    if wall_collision:
        return WALL_COLLISION_PENALTY
    if self_collision:
        return SELF_COLLISION_PENALTY
    if ate_food:
        return FOOD_REWARD

    # Encourage movement toward food
    if new_distance < prev_distance:
        return CLOSER_REWARD
    elif new_distance > prev_distance:
        return FARTHER_PENALTY

    return SURVIVAL_REWARD

def run_episode(mode, epsilon, train=False):
    state = game.reset()
    done = False
    steps = 0
    total_reward = 0.0

    max_steps = BASE_MAX_STEPS

    while not done and steps < max_steps:
        if mode == "demo" or train:
            action_idx = ai_model.get_action(state.flatten(), epsilon, device)
            action = Action(action_idx)
        else:
            action = Action.STRAIGHT
            action_idx = 0

        wall_collision, self_collision, ate_food = game.step(action)
        next_state = game.get_state()

        done = wall_collision or self_collision
        
        max_steps = BASE_MAX_STEPS + game.score * EXTRA_STEPS_PER_FOOD

        reward = calculate_reward(state, next_state, wall_collision, self_collision, ate_food) + SURVIVAL_REWARD

        total_reward += reward

        if mode == "training":
            ai_model.remember(state.flatten(), action_idx, reward, next_state.flatten(), done)

            if steps % 5 == 0:
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

    return game.score, steps, total_reward, False

def create_new_project():
    """Create a new project with user-selected location via file dialog."""
    global current_project, best_model_path, checkpoint_path, models_dir, current_model_info, episode, best_reward, best_score
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Get user data directory for default location
        try:
            from .utils import get_user_data_dir
        except ImportError:
            from utils import get_user_data_dir
        
        user_data_dir = get_user_data_dir()
        projects_dir = os.path.join(user_data_dir, "projects")
        os.makedirs(projects_dir, exist_ok=True)
        
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
            
            # Try to load the best model if it exists
            if current_project.best_model_path.exists():
                success, container = load_unified_model(best_model_path, ai_model.model, ai_model.optimizer, device=device)
                if success and container:
                    print(f"✓ Loaded best model from project")
                else:
                    print("No valid model found in project - will train from scratch")
            
            # Try to load checkpoint for training continuation
            episode = current_project.metadata.get("total_episodes", 0)
            best_reward = current_project.metadata.get("best_reward", float('-inf'))
            best_score = current_project.metadata.get("best_score", 0)
            
            if current_project.checkpoint_path.exists():
                try:
                    success, container = load_unified_model(checkpoint_path, ai_model.model, ai_model.optimizer, device=device)
                    if success and container:
                        episode = container.episode
                        best_score = container.score
                        best_reward = container.total_reward if hasattr(container, 'total_reward') else container.best_reward
                        print(f"Resumed from checkpoint - Episode: {episode}, Best Score: {best_score}, Best Reward: {best_reward:.2f}")
                    else:
                        print("No valid checkpoint found - will start from loaded model")
                except Exception as e:
                    print(f"Checkpoint loading failed: {e}")
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
epsilon_end = 0.05
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
        decay_rate = -math.log(epsilon_end / epsilon_start) / training_iterations
        training_start_time = time.time()  # Track total training time
        training_session_episodes = 0  # Track episodes in this session

        for i in range(training_iterations):
            epsilon = max(epsilon_end, epsilon_start - i / (training_iterations / 2))
            start_time = time.time()
            score, steps, total_reward, _ = run_episode(mode, epsilon=epsilon, train=True)
            ai_model.train(device)

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
        
        playing = True
        while playing:
            start_time = time.time()
            score, steps, total_reward, esc_pressed = run_episode(mode=mode, epsilon=0.0)  # No exploration in demo
            episode += 1
            print(f"AI Demo Episode: {episode}, Score: {score}, Steps: {steps}, Total Reward: {total_reward:.2f}, Duration: {time.time() - start_time:.2f}s")

            if esc_pressed:
                resume_requested = False
                playing = False

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
