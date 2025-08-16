"""
TensorBoard integration for Snake AI strategy testing and visualization.
"""

import os
import time
from typing import Dict, List, Any
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

class SnakeAITensorBoard:
    """TensorBoard logger for Snake AI experiments."""
    
    def __init__(self, log_dir: str = "runs", experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"snake_ai_{int(time.time())}"
        self.full_log_dir = os.path.join(log_dir, self.experiment_name)
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.full_log_dir)
        else:
            self.writer = None
            
        self.episode_data = []
        
    def log_episode(self, episode: int, strategy_name: str, score: int, 
                   reward: float, steps: int, epsilon: float, 
                   reward_components: Dict[str, float] = None):
        """Log episode data to TensorBoard."""
        if not self.writer:
            return
            
        # Main metrics
        self.writer.add_scalar(f'Performance/Score', score, episode)
        self.writer.add_scalar(f'Performance/Total_Reward', reward, episode)
        self.writer.add_scalar(f'Performance/Steps', steps, episode)
        self.writer.add_scalar(f'Training/Epsilon', epsilon, episode)
        
        # Strategy-specific metrics
        self.writer.add_scalar(f'Strategy/{strategy_name}/Score', score, episode)
        self.writer.add_scalar(f'Strategy/{strategy_name}/Reward', reward, episode)
        
        # Reward component breakdown
        if reward_components:
            for component, value in reward_components.items():
                self.writer.add_scalar(f'Rewards/{component}', value, episode)
                self.writer.add_scalar(f'Strategy_Rewards/{strategy_name}/{component}', value, episode)
        
        # Store for comparative analysis
        self.episode_data.append({
            'episode': episode,
            'strategy': strategy_name,
            'score': score,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'reward_components': reward_components or {}
        })
    
    def log_strategy_comparison(self, episode: int, strategies_data: Dict[str, Dict]):
        """Log comparative data between strategies."""
        if not self.writer:
            return
            
        for strategy_name, data in strategies_data.items():
            for metric, value in data.items():
                self.writer.add_scalar(f'Comparison/{strategy_name}/{metric}', value, episode)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and final metrics."""
        if not self.writer:
            return
            
        self.writer.add_hparams(hparams, metrics)
    
    def log_reward_heatmap(self, episode: int, reward_weights: Dict[str, float], 
                          performance_score: float):
        """Create heatmap of reward weights vs performance."""
        if not self.writer:
            return
            
        # Create a grid visualization of reward weights
        weight_names = list(reward_weights.keys())
        weight_values = list(reward_weights.values())
        
        # Log as individual scalars for now (TensorBoard heatmaps need special formatting)
        for name, value in reward_weights.items():
            self.writer.add_scalar(f'Reward_Weights/{name}', value, episode)
        
        self.writer.add_scalar('Performance_vs_Weights/Score', performance_score, episode)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def get_experiment_url(self) -> str:
        """Get the TensorBoard URL for this experiment."""
        return f"http://localhost:6006/#scalars&regexInput={self.experiment_name}"


class StrategyTestBed:
    """Test bed for comparing different strategies and reward systems."""
    
    def __init__(self, log_dir: str = "strategy_experiments"):
        self.log_dir = log_dir
        self.experiments = {}
        
    def create_experiment(self, name: str, description: str = "") -> SnakeAITensorBoard:
        """Create a new experiment."""
        logger = SnakeAITensorBoard(self.log_dir, name)
        self.experiments[name] = {
            'logger': logger,
            'description': description,
            'created': time.time()
        }
        return logger
    
    def run_strategy_comparison(self, strategies: List[str], reward_configs: List[Dict], 
                              episodes_per_test: int = 100):
        """Run comparative testing between strategies and reward configurations."""
        results = {}
        
        for strategy in strategies:
            for i, reward_config in enumerate(reward_configs):
                experiment_name = f"{strategy}_config_{i}"
                logger = self.create_experiment(
                    experiment_name, 
                    f"Testing {strategy} with reward config {i}"
                )
                
                # This would integrate with your existing training loop
                print(f"Running experiment: {experiment_name}")
                print(f"Strategy: {strategy}")
                print(f"Reward config: {reward_config}")
                
                # Store experiment configuration
                results[experiment_name] = {
                    'strategy': strategy,
                    'reward_config': reward_config,
                    'logger': logger
                }
        
        return results
    
    def generate_comparison_report(self) -> str:
        """Generate a comparative analysis report."""
        report = "# Snake AI Strategy Comparison Report\n\n"
        
        for exp_name, exp_data in self.experiments.items():
            report += f"## Experiment: {exp_name}\n"
            report += f"Description: {exp_data['description']}\n"
            report += f"TensorBoard URL: {exp_data['logger'].get_experiment_url()}\n\n"
        
        return report


# Integration with existing Snake AI framework
def integrate_tensorboard_with_training_loop():
    """Example of how to integrate TensorBoard with your existing training loop."""
    
    example_code = '''
    # In your main training loop (__main__.py), add:
    
    from tensorboard_logger import SnakeAITensorBoard
    
    # Initialize logger
    tb_logger = SnakeAITensorBoard("experiments", f"training_{strategy_choice}")
    
    # In your training loop:
    for i in range(training_iterations):
        # ... existing training code ...
        
        # Log episode data
        tb_logger.log_episode(
            episode=episode + i,
            strategy_name=strategy_choice,
            score=score,
            reward=total_reward,
            steps=steps,
            epsilon=epsilon,
            reward_components={
                'food_reward': food_rewards,
                'survival_reward': survival_rewards,
                'collision_penalty': collision_penalties,
                # ... other reward components
            }
        )
        
        # Log strategy-specific data every 100 episodes
        if i % 100 == 0:
            strategy_data = {
                strategy_choice: {
                    'avg_score': np.mean(episode_scores[-100:]),
                    'avg_reward': np.mean(episode_rewards[-100:]),
                    'success_rate': success_rate
                }
            }
            tb_logger.log_strategy_comparison(episode + i, strategy_data)
    
    # Log final hyperparameters
    tb_logger.log_hyperparameters(
        {
            'strategy': strategy_choice,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            **custom_weights  # reward weights
        },
        {
            'final_avg_score': np.mean(episode_scores[-100:]),
            'best_score': max(episode_scores),
            'final_avg_reward': np.mean(episode_rewards[-100:])
        }
    )
    
    tb_logger.close()
    '''
    
    return example_code


if __name__ == "__main__":
    # Example usage
    testbed = StrategyTestBed()
    
    # Define strategies and reward configurations to test
    strategies = ['survival', 'food_seeking', 'advanced']
    reward_configs = [
        {'food_reward': 100, 'survival_reward': 0.02, 'collision_penalty': -10},
        {'food_reward': 50, 'survival_reward': 0.05, 'collision_penalty': -5},
        {'food_reward': 200, 'survival_reward': 0.01, 'collision_penalty': -20}
    ]
    
    # Run comparative experiments
    results = testbed.run_strategy_comparison(strategies, reward_configs)
    
    print("Experiments set up. Run TensorBoard with:")
    print("tensorboard --logdir=strategy_experiments")
    print("Then open http://localhost:6006")
