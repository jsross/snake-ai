import pygame
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
                        iterations = get_training_iterations(display, font, game)
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
