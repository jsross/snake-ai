import pygame
import sys

# Menu mode constants
AI_DEMO = "ai"
MANUAL_MODE = "manual"
TRAINING_MODE = "training"
LOAD_BEST_MODEL = "load"
RESUME = "resume"
QUIT = "quit"


def choose_mode_pygame(display, font, game):
    """
    Display a menu for selecting game mode and return the selected mode and iterations.
    
    Args:
        display: Pygame display surface
        font: Pygame font object
        game: SnakeGame instance for getting window dimensions
        
    Returns:
        tuple: (selected_mode, iterations) where iterations is None unless training mode is selected
    """
    modes = {
        AI_DEMO: 'AI Demo',
        MANUAL_MODE: 'Manual Mode',
        TRAINING_MODE: 'Training Mode',
        LOAD_BEST_MODEL: 'Load Best Model',
        RESUME: 'Resume',
        QUIT: 'Quit'
    }
    
    mode_keys = list(modes.keys())
    selected_index = 0

    while True:
        display.fill((0, 0, 0))

        for i, mode_key in enumerate(mode_keys):
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
