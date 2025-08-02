import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys

# Constants
BLOCK_SIZE = 20
SPEED = 40
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Direction Enum
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

# Point tuple
Point = namedtuple('Point', 'x, y')

# Action Enum
class Action(Enum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2

# SnakeGame class
class SnakeGame:
    """
    A class to represent the Snake game using only the grid for state.
    """

    def __init__(self, w=640, h=480, initial_snake_length=5):
        self.w = w
        self.h = h
        self.grid_width = self.w // BLOCK_SIZE
        self.grid_height = self.h // BLOCK_SIZE
        self.initial_snake_length = max(3, initial_snake_length)  # Minimum length of 3
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.score = 0
        self.frame_iteration = 0
        # Initialize grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        # Place snake in the center with configurable length
        mid_x = self.grid_width // 2
        mid_y = self.grid_height // 2
        
        # Ensure snake fits in the grid
        max_length = min(self.initial_snake_length, mid_x)
        
        # Place snake horizontally, head at center, body extending left
        for i in range(max_length):
            self.grid[mid_y, mid_x - i] = i + 1  # head=1, body=2,3,4...
        
        self._place_food()
        
        return self.grid.copy()

    def _place_food(self):
        # Place food at a random empty cell, with random tail extension
        empty = np.argwhere(self.grid == 0)
        if len(empty) == 0:
            return
        y, x = empty[random.randint(0, len(empty) - 1)]
        tail_extension = random.choice([1, 2, 3, 5, 10])
        self.grid[y, x] = -tail_extension

    def get_state(self):
        return self.grid.copy()

    def get_cell_count(self):
        return self.grid_width * self.grid_height

    def step(self, action: Action):
        self.frame_iteration += 1
        # Event handling and quitting is now managed in main

        # Find head
        head_pos = np.argwhere(self.grid == 1)
        if len(head_pos) == 0:
            raise Exception("No head found in grid!")
        
        y, x = head_pos[0]
        # Determine new direction
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == Action.STRAIGHT:
            new_dir = clockwise[idx]
        elif action == Action.RIGHT:
            new_dir = clockwise[(idx + 1) % 4]
        elif action == Action.LEFT:
            new_dir = clockwise[(idx - 1) % 4]
        else:
            raise ValueError("Invalid action")

        self.direction = new_dir

        # Move head
        if self.direction == Direction.RIGHT:
            nx, ny = x + 1, y
        elif self.direction == Direction.LEFT:
            nx, ny = x - 1, y
        elif self.direction == Direction.DOWN:
            nx, ny = x, y + 1
        elif self.direction == Direction.UP:
            nx, ny = x, y - 1
        else:
            raise ValueError("Invalid direction")

        # Check collision
        wall_collision = False
        self_collision = False
        
        if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
            wall_collision = True
        elif self.grid[ny, nx] > 0:  # Collision with self
            self_collision = True
        
        # Return early if any collision occurred
        if wall_collision or self_collision:
            return wall_collision, self_collision, False

        # Check food
        grow = 0

        if self.grid[ny, nx] < 0:
            grow = abs(self.grid[ny, nx])
            self.score += 1
            self._place_food()

        # Decrement all positive values (body)
        mask = self.grid > 0
        self.grid[mask] += 1  # increment all body parts
        # Set new head
        self.grid[ny, nx] = 1
        # Remove tail unless growing
        if grow == 0:
            # Remove the cell with the highest value (tail) 
            tail_val = self.grid.max()
            tail_pos = np.argwhere(self.grid == tail_val)
            if len(tail_pos) > 0:
                ty, tx = tail_pos[0]
                self.grid[ty, tx] = 0
        else:
            # Do not remove tail for grow steps (tail will be further away)
            pass
        
        return False, False, grow > 0
