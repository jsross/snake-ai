"""
OpenAI Gym environment wrapper for Snake AI.
This creates a standardized interface for testing with various RL libraries.
"""

import numpy as np
import gym
from gym import spaces
import pygame

try:
    from .snake_game import SnakeGame, Action
    from .strategies import TrainingStrategy
except ImportError:
    # Fallback imports
    from snake_game import SnakeGame, Action
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))


class SnakeGymEnv(gym.Env):
    """
    OpenAI Gym environment for Snake AI.
    
    This wrapper allows you to use your Snake game with any Gym-compatible
    RL library like Stable-Baselines3, Ray RLlib, etc.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, width=20, height=20, initial_snake_length=3, 
                 reward_strategy=None, render_mode=None):
        super().__init__()
        
        # Initialize the Snake game
        self.game = SnakeGame(width, height, initial_snake_length)
        self.reward_strategy = reward_strategy
        self.render_mode = render_mode
        
        # Define action space (4 possible actions: up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (game grid)
        # Values: 0 = empty, 1 = snake body, 2 = snake head, -1 = food
        self.observation_space = spaces.Box(
            low=-1, high=2, 
            shape=(height, width), 
            dtype=np.int32
        )
        
        # Pygame setup for rendering
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((width * 20, height * 20))
            pygame.display.set_caption('Snake Gym Environment')
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        
        self.reset()
    
    def step(self, action):
        """Execute one time step within the environment."""
        # Convert gym action to game action
        action_map = {0: Action.UP, 1: Action.DOWN, 2: Action.LEFT, 3: Action.RIGHT}
        game_action = action_map[action]
        
        # Store previous state for reward calculation
        prev_score = self.game.score
        prev_head_pos = self.game.snake[0] if self.game.snake else (0, 0)
        prev_food_distance = self._calculate_food_distance()
        
        # Execute action
        reward, done, info = self.game.play_step(game_action)
        
        # Calculate enhanced reward if strategy is provided
        if self.reward_strategy:
            reward = self._calculate_strategy_reward(prev_score, prev_head_pos, prev_food_distance)
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info.update({
            'score': self.game.score,
            'snake_length': len(self.game.snake),
            'food_position': self.game.food,
            'steps': getattr(self.game, 'steps', 0)
        })
        
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment to an initial state and returns an initial observation."""
        self.game.reset()
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human' and self.screen:
            self._render_pygame()
        elif mode == 'rgb_array':
            return self._get_rgb_array()
        else:
            # Text-based rendering
            return self._render_text()
    
    def close(self):
        """Close the environment."""
        if self.screen:
            pygame.quit()
    
    def _get_observation(self):
        """Get the current state as a numpy array."""
        return self.game.get_state()
    
    def _calculate_food_distance(self):
        """Calculate Manhattan distance to food."""
        if not self.game.snake or not self.game.food:
            return 0
        head = self.game.snake[0]
        return abs(head[0] - self.game.food[0]) + abs(head[1] - self.game.food[1])
    
    def _calculate_strategy_reward(self, prev_score, prev_head_pos, prev_food_distance):
        """Calculate reward using the provided strategy."""
        if not self.reward_strategy:
            return 0
        
        # Create context for strategy
        current_distance = self._calculate_food_distance()
        
        reward_components = {
            'score_change': self.game.score - prev_score,
            'distance_change': prev_food_distance - current_distance,
            'collision': self.game.collision,
            'food_eaten': self.game.score > prev_score
        }
        
        return self.reward_strategy.calculate_reward(**reward_components)
    
    def _render_pygame(self):
        """Render using pygame."""
        if not self.screen:
            return
        
        # Fill screen with black
        self.screen.fill((0, 0, 0))
        
        # Draw game elements
        cell_size = 20
        
        # Draw snake
        for segment in self.game.snake:
            rect = pygame.Rect(segment[0] * cell_size, segment[1] * cell_size, 
                             cell_size, cell_size)
            pygame.draw.rect(self.screen, (0, 255, 0), rect)
        
        # Draw food
        if self.game.food:
            food_rect = pygame.Rect(self.game.food[0] * cell_size, 
                                  self.game.food[1] * cell_size, 
                                  cell_size, cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
        
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS for human viewing
    
    def _get_rgb_array(self):
        """Get RGB array for video recording."""
        if not self.screen:
            # Create a simple RGB representation
            height, width = self.game.get_state().shape
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            state = self.game.get_state()
            # Snake body = green
            rgb_array[state == 1] = [0, 255, 0]
            # Snake head = bright green  
            rgb_array[state == 2] = [0, 255, 100]
            # Food = red
            rgb_array[state == -1] = [255, 0, 0]
            
            return rgb_array
        else:
            # Get from pygame surface
            return pygame.surfarray.array3d(self.screen)
    
    def _render_text(self):
        """Simple text representation."""
        state = self.game.get_state()
        text = ""
        for row in state:
            for cell in row:
                if cell == -1:  # Food
                    text += "F "
                elif cell == 2:  # Snake head
                    text += "H "
                elif cell == 1:  # Snake body
                    text += "S "
                else:  # Empty
                    text += ". "
            text += "\n"
        return f"Score: {self.game.score}\n{text}"


class SnakeGymTestBed:
    """Test bed for running experiments with the Gym environment."""
    
    def __init__(self):
        self.results = {}
    
    def test_random_policy(self, episodes=100, render=False):
        """Test random policy for baseline."""
        env = SnakeGymEnv(render_mode='human' if render else None)
        
        scores = []
        for episode in range(episodes):
            observation = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                action = env.action_space.sample()  # Random action
                observation, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
            
            scores.append(info['score'])
            
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                print(f"Episode {episode}: Avg Score = {avg_score:.2f}")
        
        env.close()
        return {
            'avg_score': np.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'std_score': np.std(scores)
        }
    
    def test_strategy_policy(self, strategy, episodes=100, render=False):
        """Test a specific strategy."""
        env = SnakeGymEnv(reward_strategy=strategy, render_mode='human' if render else None)
        
        scores = []
        rewards = []
        
        for episode in range(episodes):
            observation = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Use strategy to determine action (simplified)
                action = self._strategy_to_action(strategy, observation, env.game)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
            
            scores.append(info['score'])
            rewards.append(total_reward)
            
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_reward = np.mean(rewards[-10:])
                print(f"Episode {episode}: Avg Score = {avg_score:.2f}, Avg Reward = {avg_reward:.2f}")
        
        env.close()
        return {
            'avg_score': np.mean(scores),
            'avg_reward': np.mean(rewards),
            'max_score': max(scores),
            'strategy_name': strategy.__class__.__name__ if hasattr(strategy, '__class__') else str(strategy)
        }
    
    def _strategy_to_action(self, strategy, observation, game):
        """Convert strategy decision to gym action (simplified)."""
        # This is a simplified conversion - you'd need to implement
        # proper strategy-to-action mapping based on your strategy classes
        
        if not game.snake or not game.food:
            return 0  # Default action
        
        head = game.snake[0]
        food = game.food
        
        # Simple food-seeking behavior
        if food[0] > head[0]:  # Food is to the right
            return 3  # RIGHT
        elif food[0] < head[0]:  # Food is to the left
            return 2  # LEFT
        elif food[1] > head[1]:  # Food is down
            return 1  # DOWN
        else:  # Food is up
            return 0  # UP
    
    def compare_strategies(self, strategies, episodes_per_strategy=100):
        """Compare multiple strategies."""
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting strategy: {strategy}")
            results[str(strategy)] = self.test_strategy_policy(strategy, episodes_per_strategy)
        
        return results
    
    def run_hyperparameter_sweep(self, strategy_class, param_ranges, episodes=50):
        """Run hyperparameter sweep for a strategy."""
        results = []
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Create strategy with these parameters
            strategy = strategy_class(**params)
            
            # Test strategy
            result = self.test_strategy_policy(strategy, episodes)
            result['parameters'] = params
            results.append(result)
            
            print(f"Params: {params}, Avg Score: {result['avg_score']:.2f}")
        
        return results


# Integration example
def create_stable_baselines_env():
    """Create environment compatible with Stable-Baselines3."""
    example_code = '''
    # Install: pip install stable-baselines3
    
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from snake_gym_env import SnakeGymEnv
    
    # Create environment
    def make_env():
        return SnakeGymEnv(width=10, height=10)
    
    env = DummyVecEnv([make_env])
    
    # Train with PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Test the trained model
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    '''
    return example_code


if __name__ == "__main__":
    # Example usage
    testbed = SnakeGymTestBed()
    
    print("Testing random policy...")
    random_results = testbed.test_random_policy(episodes=50, render=False)
    print(f"Random policy results: {random_results}")
    
    print("\nGym environment ready for RL experiments!")
    print("You can now use this with:")
    print("1. Stable-Baselines3")
    print("2. Ray RLlib") 
    print("3. OpenAI Spinning Up")
    print("4. Any other Gym-compatible library")
