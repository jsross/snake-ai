import pygame
import numpy as np
import io
import base64

BLOCK_SIZE = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (192, 192, 192)

def render(display, grid, score, font):
    display.fill(BLACK)
    grid_height, grid_width = grid.shape
    for y in range(grid_height):
        for x in range(grid_width):
            val = grid[y, x]
            if val > 0:
                pygame.draw.rect(display, GREEN, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(display, BLUE, pygame.Rect(x * BLOCK_SIZE + 4, y * BLOCK_SIZE + 4, 12, 12))
            elif val < 0:
                pygame.draw.rect(display, RED, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    text = font.render(f"Score: {score}", True, WHITE)
    display.blit(text, [0, 0])
    pygame.display.flip()

def render_training_progress(display, font, episode, total_episodes, current_score, best_score, strategy_name, eta_str=""):
    """Render training progress in the pygame window."""
    display.fill(BLACK)
    
    # Calculate progress
    progress = episode / total_episodes if total_episodes > 0 else 0
    
    # Window dimensions
    width, height = display.get_size()
    
    # Title
    title_text = font.render("Training in Progress...", True, WHITE)
    title_rect = title_text.get_rect(center=(width//2, 50))
    display.blit(title_text, title_rect)
    
    # Strategy name
    strategy_text = font.render(f"Strategy: {strategy_name}", True, WHITE)
    strategy_rect = strategy_text.get_rect(center=(width//2, 90))
    display.blit(strategy_text, strategy_rect)
    
    # Progress bar
    bar_width = width - 100
    bar_height = 30
    bar_x = 50
    bar_y = 150
    
    # Background bar
    pygame.draw.rect(display, GRAY, (bar_x, bar_y, bar_width, bar_height))
    
    # Progress fill
    fill_width = int(bar_width * progress)
    pygame.draw.rect(display, GREEN, (bar_x, bar_y, fill_width, bar_height))
    
    # Progress text
    progress_text = font.render(f"{progress*100:.1f}% ({episode}/{total_episodes})", True, WHITE)
    progress_rect = progress_text.get_rect(center=(width//2, bar_y + bar_height + 30))
    display.blit(progress_text, progress_rect)
    
    # Stats
    score_text = font.render(f"Current Score: {current_score}", True, WHITE)
    score_rect = score_text.get_rect(center=(width//2, 230))
    display.blit(score_text, score_rect)
    
    best_text = font.render(f"Best Score: {best_score}", True, WHITE)
    best_rect = best_text.get_rect(center=(width//2, 260))
    display.blit(best_text, best_rect)
    
    # ETA
    if eta_str:
        eta_text = font.render(eta_str, True, WHITE)
        eta_rect = eta_text.get_rect(center=(width//2, 290))
        display.blit(eta_text, eta_rect)
    
    pygame.display.flip()

def render_training_complete(display, font, episode_rewards, average_rewards, final_score, best_score, strategy_name, total_time_minutes):
    """Render training completion screen with plot in pygame window."""
    display.fill(BLACK)
    
    # Window dimensions
    width, height = display.get_size()
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as agg
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_title("Training Complete - Results", color='white', fontsize=14)
        ax.set_xlabel("Episode", color='white')
        ax.set_ylabel("Reward", color='white')
        ax.grid(True, alpha=0.3)
        
        # Plot data
        ax.plot(episode_rewards, label='Episode Rewards', alpha=0.3, color='lightgreen')
        ax.plot(average_rewards, label='Average Reward (100)', color='green', linewidth=2)
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        # Style the plot for dark theme
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        
        # Convert plot to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get the raw buffer data (compatible with newer matplotlib versions)
        try:
            # Try the newer method first (matplotlib >= 3.1)
            buf = canvas.buffer_rgba()
            size = canvas.get_width_height()
            plot_surface = pygame.image.frombuffer(buf, size, 'RGBA')
        except AttributeError:
            try:
                # Fallback to older method
                renderer = canvas.get_renderer()
                raw_data = renderer.tostring_rgb()
                size = canvas.get_width_height()
                plot_surface = pygame.image.fromstring(raw_data, size, 'RGB')
            except AttributeError:
                # Last resort - save to buffer and load
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', facecolor='black', edgecolor='none', bbox_inches='tight')
                buf.seek(0)
                plot_surface = pygame.image.load(buf)
                buf.close()
        
        # Scale plot to fit in window (leave space for text)
        plot_height = height - 200  # Leave space for text
        plot_width = width - 100
        plot_surface = pygame.transform.scale(plot_surface, (plot_width, plot_height))
        
        # Center the plot horizontally
        plot_x = (width - plot_width) // 2
        plot_y = 50
        
        # Draw the plot
        display.blit(plot_surface, (plot_x, plot_y))
        
        plt.close(fig)  # Clean up
        
    except Exception as e:
        # Fallback if matplotlib fails - show a simple text summary
        error_text = font.render("Training Plot Display Error", True, RED)
        error_rect = error_text.get_rect(center=(width//2, height//2 - 60))
        display.blit(error_text, error_rect)
        
        # Show basic stats instead
        if episode_rewards and average_rewards:
            final_reward = episode_rewards[-1] if episode_rewards else 0
            avg_reward = average_rewards[-1] if average_rewards else 0
            max_reward = max(episode_rewards) if episode_rewards else 0
            
            stats_y = height//2 - 20
            final_text = font.render(f"Final Episode Reward: {final_reward:.2f}", True, WHITE)
            final_rect = final_text.get_rect(center=(width//2, stats_y))
            display.blit(final_text, final_rect)
            
            avg_text = font.render(f"Average Reward (100): {avg_reward:.2f}", True, WHITE)
            avg_rect = avg_text.get_rect(center=(width//2, stats_y + 30))
            display.blit(avg_text, avg_rect)
            
            max_text = font.render(f"Best Episode Reward: {max_reward:.2f}", True, WHITE)
            max_rect = max_text.get_rect(center=(width//2, stats_y + 60))
            display.blit(max_text, max_rect)
        
        print(f"Matplotlib plot display failed: {e}")
    
    # Training summary text at bottom
    y_offset = height - 150
    
    title_text = font.render("Training Complete!", True, WHITE)
    title_rect = title_text.get_rect(center=(width//2, y_offset))
    display.blit(title_text, title_rect)
    
    strategy_text = font.render(f"Strategy: {strategy_name}", True, WHITE)
    strategy_rect = strategy_text.get_rect(center=(width//2, y_offset + 30))
    display.blit(strategy_text, strategy_rect)
    
    score_text = font.render(f"Final Score: {final_score} | Best Score: {best_score}", True, WHITE)
    score_rect = score_text.get_rect(center=(width//2, y_offset + 60))
    display.blit(score_text, score_rect)
    
    time_text = font.render(f"Training Time: {total_time_minutes:.1f} minutes", True, WHITE)
    time_rect = time_text.get_rect(center=(width//2, y_offset + 90))
    display.blit(time_text, time_rect)
    
    instruction_text = font.render("Press any key to continue...", True, GRAY)
    instruction_rect = instruction_text.get_rect(center=(width//2, y_offset + 120))
    display.blit(instruction_text, instruction_rect)
    
    pygame.display.flip() 