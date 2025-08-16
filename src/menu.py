import pygame
from pygame.locals import *
import sys

# Menu mode constants
CREATE_NEW_MODEL = "create_new"
LOAD_MODEL_INTERACTIVE = "load_interactive"
AI_DEMO = "ai"
MANUAL_MODE = "manual"
TRAINING_MODE = "training"
RESUME = "resume"
QUIT = "quit"


def choose_mode_pygame(display, font, game, current_model_info=None):
    """
    Display a menu for selecting game mode and return the selected mode and iterations.
    
    Args:
        display: Pygame display surface
        font: Pygame font object
        game: SnakeGame instance for getting window dimensions
        current_model_info: Dict with project/model status info
        
    Returns:
        tuple: (selected_mode, iterations) where iterations is None unless training mode is selected
    """
    # Determine if project/model is loaded
    model_loaded = current_model_info and current_model_info.get('loaded', False)
    
    modes = {
        CREATE_NEW_MODEL: 'Create New Project',
        LOAD_MODEL_INTERACTIVE: 'Load Project',
        AI_DEMO: 'AI Demo' + (' (requires project)' if not model_loaded else ''),
        MANUAL_MODE: 'Manual Mode',
        TRAINING_MODE: 'Training Mode' + (' (requires project)' if not model_loaded else ''),
        RESUME: 'Resume',
        QUIT: 'Quit'
    }
    
    mode_keys = list(modes.keys())
    selected_index = 0

    while True:
        display.fill((0, 0, 0))
        
        # Display current project status at the top
        if current_model_info and current_model_info.get('loaded', False):
            project_name = current_model_info.get('project_name', 'Unknown')
            episode = current_model_info.get('episode', 0)
            status_text = f"Project: {project_name} | Episode: {episode}"
            status_color = (0, 255, 0)  # Green for loaded
        else:
            status_text = "No Project Loaded - Create or Load a Project First"
            status_color = (255, 100, 100)  # Red for not loaded
        
        status_surface = font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(center=(game.w // 2, 50))
        display.blit(status_surface, status_rect)

        # Display menu options
        for i, mode_key in enumerate(mode_keys):
            # Disable AI Demo and Training if no project is loaded
            if not model_loaded and mode_key in [AI_DEMO, TRAINING_MODE]:
                color = (60, 60, 60)  # Dark gray for disabled
            else:
                color = (255, 255, 255) if i == selected_index else (100, 100, 100)
            
            text = font.render(modes[mode_key], True, color)
            rect = text.get_rect(center=(game.w // 2, game.h // 2 + i * 30))
            display.blit(text, rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(mode_keys)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(mode_keys)
                elif event.key == pygame.K_RETURN:
                    selected_key = mode_keys[selected_index]
                    
                    # Prevent selection of AI Demo and Training if no project loaded
                    if not model_loaded and selected_key in [AI_DEMO, TRAINING_MODE]:
                        continue  # Don't allow selection
                    
                    iterations = None
                    if selected_key == TRAINING_MODE:
                        # Don't ask for iterations here - handled in main flow with strategy selection
                        iterations = None
                    elif selected_key == QUIT:
                        pygame.quit()
                        sys.exit()
                    return selected_key, iterations


def get_training_iterations(display, font, game):
    """
    Get the number of training iterations from user input.
    
    Args:
        display: Pygame display surface
        font: Pygame font object
        game: SnakeGame instance for getting window dimensions
        
    Returns:
        int: Number of training iterations
    """
    iterations = ""
    entering = True
    
    while entering:
        display.fill((0, 0, 0))
        prompt_text = font.render("Enter training iterations: " + iterations, True, (255, 255, 255))
        rect = prompt_text.get_rect(center=(game.w // 2, game.h // 2))
        display.blit(prompt_text, rect)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and iterations.isdigit():
                    entering = False
                elif event.key == pygame.K_BACKSPACE:
                    iterations = iterations[:-1]
                elif event.unicode.isdigit():
                    iterations += event.unicode
    return int(iterations)


def choose_training_strategy(display, font, game):
    """
    Display a menu for selecting training strategy.
    
    Args:
        display: Pygame display surface
        font: Pygame font object
        game: SnakeGame instance for getting window dimensions
        
    Returns:
        str: Selected strategy name ('auto', 'survival', 'food_seeking', 'advanced')
    """
    strategies = {
        'auto': 'Auto (Curriculum Learning)',
        'survival': 'Survival Strategy',
        'food_seeking': 'Food Seeking Strategy', 
        'advanced': 'Advanced Strategy'
    }
    
    strategy_keys = list(strategies.keys())
    selected_index = 0

    while True:
        display.fill((0, 0, 0))
        
        # Title
        title_text = font.render("Select Training Strategy", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(game.w // 2, 50))
        display.blit(title_text, title_rect)
        
        # Strategy options
        for i, (_, description) in enumerate(strategies.items()):
            color = (255, 255, 0) if i == selected_index else (255, 255, 255)
            option_text = font.render(f"{i + 1}. {description}", True, color)
            option_rect = option_text.get_rect(center=(game.w // 2, 150 + i * 50))
            display.blit(option_text, option_rect)
        
        # Instructions
        instructions = [
            "Use UP/DOWN arrows to navigate",
            "Press ENTER to select",
            "Press ESC to go back"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = font.render(instruction, True, (128, 128, 128))
            inst_rect = inst_text.get_rect(center=(game.w // 2, 400 + i * 30))
            display.blit(inst_text, inst_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(strategy_keys)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(strategy_keys)
                elif event.key == pygame.K_RETURN:
                    return strategy_keys[selected_index]
                elif event.key == pygame.K_ESCAPE:
                    return None  # User cancelled
                elif event.key == pygame.K_1:
                    return 'auto'
                elif event.key == pygame.K_2:
                    return 'survival'
                elif event.key == pygame.K_3:
                    return 'food_seeking'
                elif event.key == pygame.K_4:
                    return 'advanced'


def configure_strategy_weights(display, font, game, strategy_name):
    """
    Allow user to configure reward weights for a strategy.
    
    Args:
        display: Pygame display surface
        font: Pygame font object
        game: SnakeGame instance for getting window dimensions
        strategy_name: Name of the strategy to configure
        
    Returns:
        dict: Updated reward weights or None if cancelled
    """
    if strategy_name == 'auto':
        return None  # Auto mode uses default weights
    
    # Import strategy classes to get default weights
    try:
        from .strategies.survival_strategy import SurvivalTrainingStrategy
        from .strategies.food_seeking_strategy import FoodSeekingTrainingStrategy
        from .strategies.advanced_strategy import AdvancedTrainingStrategy
    except ImportError:
        from strategies.survival_strategy import SurvivalTrainingStrategy
        from strategies.food_seeking_strategy import FoodSeekingTrainingStrategy
        from strategies.advanced_strategy import AdvancedTrainingStrategy
    
    strategy_classes = {
        'survival': SurvivalTrainingStrategy,
        'food_seeking': FoodSeekingTrainingStrategy,
        'advanced': AdvancedTrainingStrategy
    }
    
    if strategy_name not in strategy_classes:
        return None
        
    # Get current weights from strategy
    strategy = strategy_classes[strategy_name]()
    current_weights = strategy.get_reward_weights()
    
    # Define which weights are configurable for each strategy
    configurable_weights = {
        'survival': ['wall_collision_penalty', 'self_collision_penalty', 'survival_reward'],
        'food_seeking': ['wall_collision_penalty', 'self_collision_penalty', 'food_reward', 'closer_reward', 'farther_penalty'],
        'advanced': ['wall_collision_penalty', 'self_collision_penalty', 'food_reward', 'closer_reward', 'farther_penalty', 'move_cost']
    }
    
    weights_to_configure = configurable_weights.get(strategy_name, [])
    selected_weight = 0
    editing_value = False
    current_input = ""
    
    # Create a working copy of weights
    updated_weights = current_weights.copy()
    
    while True:
        display.fill((0, 0, 0))
        
        # Title
        title_text = font.render(f"Configure {strategy_name.replace('_', ' ').title()} Strategy Weights", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(game.w // 2, 30))
        display.blit(title_text, title_rect)
        
        # Weight configuration
        y_offset = 80
        for i, weight_name in enumerate(weights_to_configure):
            color = (255, 255, 0) if i == selected_weight and not editing_value else (255, 255, 255)
            if editing_value and i == selected_weight:
                color = (0, 255, 0)
                display_value = current_input if current_input else str(updated_weights[weight_name])
            else:
                display_value = str(updated_weights[weight_name])
                
            weight_text = font.render(f"{weight_name.replace('_', ' ').title()}: {display_value}", True, color)
            weight_rect = weight_text.get_rect(center=(game.w // 2, y_offset + i * 35))
            display.blit(weight_text, weight_rect)
        
        # Instructions
        instructions = [
            "UP/DOWN: Navigate weights",
            "ENTER: Edit selected weight" if not editing_value else "ENTER: Confirm value",
            "ESC: Cancel" if not editing_value else "ESC: Cancel edit",
            "S: Save and continue",
            "R: Reset to defaults"
        ]
        
        inst_y = y_offset + len(weights_to_configure) * 35 + 40
        for i, instruction in enumerate(instructions):
            inst_text = font.render(instruction, True, (128, 128, 128))
            inst_rect = inst_text.get_rect(center=(game.w // 2, inst_y + i * 25))
            display.blit(inst_text, inst_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if editing_value:
                    if event.key == pygame.K_RETURN and current_input:
                        try:
                            updated_weights[weights_to_configure[selected_weight]] = float(current_input)
                            editing_value = False
                            current_input = ""
                        except ValueError:
                            current_input = ""  # Invalid input, clear
                    elif event.key == pygame.K_ESCAPE:
                        editing_value = False
                        current_input = ""
                    elif event.key == pygame.K_BACKSPACE:
                        current_input = current_input[:-1]
                    elif event.key == pygame.K_MINUS:
                        current_input += "-"
                    elif event.unicode.replace('.', '').replace('-', '').isdigit() or event.unicode == '.':
                        current_input += event.unicode
                else:
                    if event.key == pygame.K_UP:
                        selected_weight = (selected_weight - 1) % len(weights_to_configure)
                    elif event.key == pygame.K_DOWN:
                        selected_weight = (selected_weight + 1) % len(weights_to_configure)
                    elif event.key == pygame.K_RETURN:
                        editing_value = True
                        current_input = str(updated_weights[weights_to_configure[selected_weight]])
                    elif event.key == pygame.K_ESCAPE:
                        return None  # User cancelled
                    elif event.key == pygame.K_s:
                        return updated_weights  # Save and continue
                    elif event.key == pygame.K_r:
                        # Reset to defaults
                        updated_weights = current_weights.copy()
