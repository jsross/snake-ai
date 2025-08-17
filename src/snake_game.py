import random
from enum import Enum
from collections import namedtuple
import numpy as np

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
        
        return self.get_features()

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
    
    def get_features(self):
        """Extract meaningful features from the game state instead of using raw grid."""
        # Find head position
        head_pos = np.argwhere(self.grid == 1)
        if len(head_pos) == 0:
            return np.zeros(20)  # Return default features if no head found
        
        head_y, head_x = head_pos[0]
        
        # Find food positions
        food_positions = np.argwhere(self.grid < 0)
        
        # Find body positions (excluding head)
        body_positions = np.argwhere(self.grid > 1)
        
        features = []
        
        # 1. Head position (normalized)
        features.extend([
            head_x / self.grid_width,    # Head X position (0-1)
            head_y / self.grid_height    # Head Y position (0-1)
        ])
        
        # 2. Direction to closest food
        if len(food_positions) > 0:
            distances = [abs(head_x - fx) + abs(head_y - fy) for fy, fx in food_positions]
            closest_food_idx = np.argmin(distances)
            food_y, food_x = food_positions[closest_food_idx]
            
            # Normalized direction to food
            food_dir_x = (food_x - head_x) / self.grid_width
            food_dir_y = (food_y - head_y) / self.grid_height
            food_distance = distances[closest_food_idx] / (self.grid_width + self.grid_height)
        else:
            food_dir_x = food_dir_y = food_distance = 0
        
        features.extend([food_dir_x, food_dir_y, food_distance])
        
        # 3. Danger detection in each direction (up, down, left, right)
        dangers = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dy, dx in directions:
            # Check immediate danger (next cell)
            next_y, next_x = head_y + dy, head_x + dx
            immediate_danger = 0
            
            # Wall collision
            if next_x < 0 or next_x >= self.grid_width or next_y < 0 or next_y >= self.grid_height:
                immediate_danger = 1
            # Body collision
            elif self.grid[next_y, next_x] > 1:
                immediate_danger = 1
            
            dangers.append(immediate_danger)
            
            # Check danger in next 3 steps
            future_danger = 0
            for step in range(2, 4):  # Check 2-3 steps ahead
                check_y, check_x = head_y + dy * step, head_x + dx * step
                if (check_x < 0 or check_x >= self.grid_width or 
                    check_y < 0 or check_y >= self.grid_height or
                    self.grid[check_y, check_x] > 1):
                    future_danger = 1 / step  # Weight closer dangers more heavily
                    break
            dangers.append(future_danger)
        
        features.extend(dangers)  # 8 features: 4 immediate + 4 future dangers
        
        # 4. Snake body information
        snake_length = len(body_positions) + 1  # +1 for head
        normalized_length = snake_length / (self.grid_width * self.grid_height)
        features.append(normalized_length)
        
        # 5. Available space in each direction
        space_counts = []
        for dy, dx in directions:
            space_count = 0
            for step in range(1, max(self.grid_width, self.grid_height)):
                check_y, check_x = head_y + dy * step, head_x + dx * step
                if (check_x < 0 or check_x >= self.grid_width or 
                    check_y < 0 or check_y >= self.grid_height or
                    self.grid[check_y, check_x] > 1):
                    break
                space_count += 1
            space_counts.append(space_count / max(self.grid_width, self.grid_height))
        
        features.extend(space_counts)  # 4 features: space in each direction
        
        # 6. Food count and distribution
        total_food = len(food_positions)
        features.append(total_food / 10)  # Normalize assuming max ~10 food items
        
        # Convert to numpy array and ensure we have exactly 20 features
        features = np.array(features[:20])
        if len(features) < 20:
            # Pad with zeros if we have fewer features
            padding = np.zeros(20 - len(features))
            features = np.concatenate([features, padding])
        
        return features
    
    def get_feature_count(self):
        """Return the number of features extracted by get_features()"""
        return 20

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
