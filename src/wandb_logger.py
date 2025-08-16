"""
Weights & Biases (wandb) integration for Snake AI experiment tracking.
"""

import os
import time
from typing import Dict, List, Any, Optional
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")


class SnakeAIWandbLogger:
    """Weights & Biases logger for Snake AI experiments."""
    
    def __init__(self, project_name: str = "snake-ai-experiments", 
                 experiment_name: str = None, config: Dict = None):
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        if not WANDB_AVAILABLE:
            print("wandb not available. Logging disabled.")
            self.run = None
            return
        
        # Initialize wandb run
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config or {}
        )
        
        # Define custom metrics
        if self.run:
            wandb.define_metric("episode")
            wandb.define_metric("score", step_metric="episode")
            wandb.define_metric("reward", step_metric="episode")
            wandb.define_metric("steps", step_metric="episode")
            wandb.define_metric("epsilon", step_metric="episode")
            
    def log_episode(self, episode: int, strategy_name: str, score: int, 
                   reward: float, steps: int, epsilon: float, 
                   reward_components: Dict[str, float] = None):
        """Log episode data to wandb."""
        if not self.run:
            return
        
        log_data = {
            "episode": episode,
            "strategy": strategy_name,
            "score": score,
            "total_reward": reward,
            "steps": steps,
            "epsilon": epsilon
        }
        
        # Add reward components
        if reward_components:
            for component, value in reward_components.items():
                log_data[f"reward_{component}"] = value
        
        wandb.log(log_data)
    
    def log_training_session(self, session_data: Dict):
        """Log complete training session results."""
        if not self.run:
            return
        
        wandb.log({
            "session_final_score": session_data.get('final_score', 0),
            "session_best_score": session_data.get('best_score', 0),
            "session_avg_score": session_data.get('avg_score', 0),
            "session_total_episodes": session_data.get('total_episodes', 0),
            "session_duration_minutes": session_data.get('duration_minutes', 0)
        })
    
    def log_strategy_comparison(self, comparison_data: Dict[str, Dict]):
        """Log strategy comparison results."""
        if not self.run:
            return
        
        # Create comparison table
        table_data = []
        for strategy, metrics in comparison_data.items():
            row = {"strategy": strategy, **metrics}
            table_data.append(row)
        
        table = wandb.Table(
            columns=["strategy"] + list(next(iter(comparison_data.values())).keys()),
            data=[[row[col] for col in table.columns] for row in table_data]
        )
        
        wandb.log({"strategy_comparison": table})
    
    def log_hyperparameter_sweep(self, sweep_results: List[Dict]):
        """Log hyperparameter sweep results."""
        if not self.run:
            return
        
        # Create sweep results table
        columns = ["avg_score", "max_score", "parameters"]
        data = []
        
        for result in sweep_results:
            data.append([
                result.get('avg_score', 0),
                result.get('max_score', 0),
                json.dumps(result.get('parameters', {}))
            ])
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"hyperparameter_sweep": table})
    
    def log_reward_analysis(self, reward_weights: Dict[str, float], 
                           performance_metrics: Dict[str, float]):
        """Log reward weight analysis."""
        if not self.run:
            return
        
        # Log reward weights
        for weight_name, weight_value in reward_weights.items():
            wandb.log({f"reward_weight_{weight_name}": weight_value})
        
        # Log performance metrics
        for metric_name, metric_value in performance_metrics.items():
            wandb.log({f"performance_{metric_name}": metric_value})
    
    def save_model_artifact(self, model_path: str, model_name: str = "snake_model"):
        """Save model as wandb artifact."""
        if not self.run:
            return
        
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish the wandb run."""
        if self.run:
            wandb.finish()


class WandbExperimentManager:
    """Manager for running multiple experiments with wandb."""
    
    def __init__(self, project_name: str = "snake-ai-experiments"):
        self.project_name = project_name
        self.experiments = {}
    
    def create_sweep_config(self, strategy_name: str, parameter_ranges: Dict) -> Dict:
        """Create wandb sweep configuration."""
        sweep_config = {
            'method': 'grid',  # or 'random', 'bayes'
            'metric': {
                'name': 'session_avg_score',
                'goal': 'maximize'
            },
            'parameters': {}
        }
        
        # Convert parameter ranges to wandb format
        for param_name, param_range in parameter_ranges.items():
            if isinstance(param_range, list):
                sweep_config['parameters'][param_name] = {'values': param_range}
            elif isinstance(param_range, dict):
                sweep_config['parameters'][param_name] = param_range
        
        return sweep_config
    
    def run_hyperparameter_sweep(self, strategy_class, parameter_ranges: Dict, 
                                count: int = 10):
        """Run hyperparameter sweep using wandb."""
        if not WANDB_AVAILABLE:
            print("wandb not available for sweep")
            return
        
        sweep_config = self.create_sweep_config(
            strategy_class.__name__, 
            parameter_ranges
        )
        
        # Create sweep
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        
        def train_with_config():
            """Training function for sweep."""
            with wandb.init() as run:
                config = wandb.config
                
                # Create strategy with sweep parameters
                strategy = strategy_class(**dict(config))
                
                # Run training (integrate with your training loop)
                logger = SnakeAIWandbLogger(
                    project_name=self.project_name,
                    experiment_name=f"sweep_{strategy_class.__name__}",
                    config=dict(config)
                )
                
                # Your training loop would go here
                # For now, simulate some results
                import random
                final_score = random.uniform(0, 50)
                avg_score = random.uniform(0, final_score)
                
                logger.log_training_session({
                    'final_score': final_score,
                    'avg_score': avg_score,
                    'total_episodes': 1000
                })
                
                logger.finish()
        
        # Run sweep
        wandb.agent(sweep_id, train_with_config, count=count)
        
        return sweep_id
    
    def compare_strategies(self, strategies: List, episodes_per_strategy: int = 100):
        """Run strategy comparison experiment."""
        comparison_results = {}
        
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            
            logger = SnakeAIWandbLogger(
                project_name=self.project_name,
                experiment_name=f"comparison_{strategy_name}",
                config={
                    'strategy': strategy_name,
                    'episodes': episodes_per_strategy
                }
            )
            
            # Run training for this strategy
            # (integrate with your training loop)
            
            # Simulate results for now
            import random
            results = {
                'avg_score': random.uniform(5, 25),
                'max_score': random.uniform(25, 50),
                'success_rate': random.uniform(0.1, 0.8)
            }
            
            comparison_results[strategy_name] = results
            
            logger.log_training_session({
                'avg_score': results['avg_score'],
                'best_score': results['max_score']
            })
            
            logger.finish()
        
        # Log final comparison
        comparison_logger = SnakeAIWandbLogger(
            project_name=self.project_name,
            experiment_name="strategy_comparison_summary"
        )
        comparison_logger.log_strategy_comparison(comparison_results)
        comparison_logger.finish()
        
        return comparison_results
    
    def analyze_reward_weights(self, weight_combinations: List[Dict], 
                              strategy_class, episodes: int = 50):
        """Analyze different reward weight combinations."""
        results = []
        
        for i, weights in enumerate(weight_combinations):
            logger = SnakeAIWandbLogger(
                project_name=self.project_name,
                experiment_name=f"reward_analysis_{i}",
                config=weights
            )
            
            # Create strategy with these weights
            strategy = strategy_class()
            # Apply weights to strategy (depends on your implementation)
            
            # Run training and collect results
            # (integrate with your training loop)
            
            # Simulate results
            import random
            performance = {
                'avg_score': random.uniform(0, 30),
                'convergence_speed': random.uniform(100, 1000),
                'stability': random.uniform(0.5, 1.0)
            }
            
            logger.log_reward_analysis(weights, performance)
            logger.finish()
            
            results.append({
                'weights': weights,
                'performance': performance
            })
        
        return results


# Integration example with your existing code
def integrate_wandb_with_training():
    """Example integration with your training loop."""
    
    integration_example = '''
    # In your main training loop (__main__.py), add:
    
    from wandb_logger import SnakeAIWandbLogger
    
    # Initialize logger
    config = {
        'strategy': strategy_choice,
        'training_iterations': training_iterations,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        **custom_weights  # reward weights
    }
    
    wandb_logger = SnakeAIWandbLogger(
        project_name="snake-ai-training",
        experiment_name=f"{strategy_choice}_{int(time.time())}",
        config=config
    )
    
    # In your training loop:
    for i in range(training_iterations):
        # ... existing training code ...
        
        # Log episode data
        reward_components = {
            'food': food_reward_total,
            'survival': survival_reward_total, 
            'collision': collision_penalty_total,
            'movement': movement_reward_total
        }
        
        wandb_logger.log_episode(
            episode=episode + i,
            strategy_name=strategy_choice,
            score=score,
            reward=total_reward,
            steps=steps,
            epsilon=epsilon,
            reward_components=reward_components
        )
    
    # Log final session results
    wandb_logger.log_training_session({
        'final_score': episode_scores[-1] if episode_scores else 0,
        'best_score': max(episode_scores) if episode_scores else 0,
        'avg_score': np.mean(episode_scores[-100:]) if episode_scores else 0,
        'total_episodes': training_iterations,
        'duration_minutes': (time.time() - training_start_time) / 60
    })
    
    # Save model as artifact
    if current_project and current_project.best_model_path.exists():
        wandb_logger.save_model_artifact(
            str(current_project.best_model_path),
            f"{strategy_choice}_best_model"
        )
    
    wandb_logger.finish()
    '''
    
    return integration_example


if __name__ == "__main__":
    # Example usage
    if WANDB_AVAILABLE:
        # Setup experiment manager
        manager = WandbExperimentManager()
        
        # Example hyperparameter sweep
        parameter_ranges = {
            'food_reward': [50, 100, 200],
            'survival_reward': [0.01, 0.02, 0.05],
            'collision_penalty': [-5, -10, -20]
        }
        
        print("Starting hyperparameter sweep...")
        # sweep_id = manager.run_hyperparameter_sweep(SomeStrategyClass, parameter_ranges)
        
        print("wandb integration ready!")
        print("1. Run: wandb login")
        print("2. Use the integration code in your training loop")
        print("3. View results at: https://wandb.ai")
    else:
        print("Install wandb to use this integration: pip install wandb")
