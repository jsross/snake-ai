import pygame
import numpy as np

BLOCK_SIZE = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

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