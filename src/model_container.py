"""
Unified Model Storage System for Snake AI

This module provides a consistent way to save and load all model-related data
including the model weights, training state, metadata, and best performance info.
"""

import torch
import os
import json
import time
from datetime import datetime

class SnakeModelContainer:
    """
    Unified container for all Snake AI model data.
    Stores model weights, training state, metadata, and performance info in a single file.
    """
    
    def __init__(self):
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.target_model_state_dict = None
        
        # Training progress
        self.episode = 0
        self.score = 0  # Current score
        self.total_reward = 0.0  # Current total reward
        self.best_reward = float('-inf')
        self.best_score = 0
        self.total_training_time = 0.0
        
        # Metadata
        self.created_date = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        self.model_version = "1.0"
        self.description = ""
        self.training_parameters = {}
        self.training_data = {}  # Additional training data
        
        # Performance history
        self.performance_history = {
            'episodes': [],
            'scores': [],
            'rewards': [],
            'timestamps': []
        }
        
        # Model architecture info
        self.architecture = {
            'input_size': None,
            'output_size': None,
            'hidden_layers': None
        }
    
    def update_from_training(self, model, optimizer, target_model, episode, score, reward, training_params=None):
        """Update container with current training state."""
        self.model_state_dict = model.state_dict().copy()
        self.optimizer_state_dict = optimizer.state_dict().copy()
        if target_model:
            self.target_model_state_dict = target_model.state_dict().copy()
        
        self.episode = episode
        self.last_updated = datetime.now().isoformat()
        
        # Update best performance
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_score = score
        
        # Add to performance history (keep last 1000 entries)
        self.performance_history['episodes'].append(episode)
        self.performance_history['scores'].append(score)
        self.performance_history['rewards'].append(reward)
        self.performance_history['timestamps'].append(time.time())
        
        # Keep only last 1000 entries to prevent file bloat
        for key in self.performance_history:
            if len(self.performance_history[key]) > 1000:
                self.performance_history[key] = self.performance_history[key][-1000:]
        
        if training_params:
            self.training_parameters.update(training_params)
    
    def load_into_model(self, model, optimizer=None, target_model=None, device=None):
        """Load the stored weights into the provided models."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        success = True
        
        try:
            if self.model_state_dict:
                # Move state dict to correct device
                state_dict = {}
                for key, value in self.model_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        state_dict[key] = value.to(device)
                    else:
                        state_dict[key] = value
                
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(f"✓ Model weights loaded (Episode: {self.episode}, Best Score: {self.best_score})")
            else:
                print("⚠ No model weights found in container")
                success = False
            
            if optimizer and self.optimizer_state_dict:
                optimizer.load_state_dict(self.optimizer_state_dict)
                print("✓ Optimizer state loaded")
            
            if target_model and self.target_model_state_dict:
                target_state_dict = {}
                for key, value in self.target_model_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        target_state_dict[key] = value.to(device)
                    else:
                        target_state_dict[key] = value
                
                target_model.load_state_dict(target_state_dict)
                target_model.to(device)
                print("✓ Target model weights loaded")
            
            return success
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
    
    def get_info(self):
        """Get human-readable information about this model."""
        info = {
            'description': self.description or 'No description',
            'created': self.created_date,
            'last_updated': self.last_updated,
            'episode': self.episode,
            'best_score': self.best_score,
            'best_reward': self.best_reward,
            'training_time_hours': self.total_training_time / 3600.0,
            'performance_entries': len(self.performance_history['episodes']),
            'architecture': self.architecture
        }
        return info
    
    def save(self, filepath):
        """Save the entire container to a file."""
        try:
            # Create directory if needed
            dir_path = os.path.dirname(filepath)
            if dir_path:  # Only create if there's actually a directory path
                os.makedirs(dir_path, exist_ok=True)
            
            # Create the data to save
            save_data = {
                'container_version': '1.0',
                'model_state_dict': self.model_state_dict,
                'optimizer_state_dict': self.optimizer_state_dict,
                'target_model_state_dict': self.target_model_state_dict,
                'episode': self.episode,
                'score': self.score,
                'total_reward': self.total_reward,
                'best_reward': self.best_reward,
                'best_score': self.best_score,
                'total_training_time': self.total_training_time,
                'created_date': self.created_date,
                'last_updated': self.last_updated,
                'model_version': self.model_version,
                'description': self.description,
                'training_parameters': self.training_parameters,
                'training_data': self.training_data,
                'performance_history': self.performance_history,
                'architecture': self.architecture
            }
            
            torch.save(save_data, filepath)
            print(f"✓ Model container saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save model container: {e}")
            return False
    
    @classmethod
    def load(cls, filepath, device=None):
        """Load a model container from a file."""
        if not os.path.exists(filepath):
            print(f"✗ Model container file not found: {filepath}")
            return None
        
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            data = torch.load(filepath, map_location=device, weights_only=False)  # Allow all content for unified containers
            
            # Create new container
            container = cls()
            
            # Load all data
            container.model_state_dict = data.get('model_state_dict')
            container.optimizer_state_dict = data.get('optimizer_state_dict')
            container.target_model_state_dict = data.get('target_model_state_dict')
            container.episode = data.get('episode', 0)
            container.score = data.get('score', 0)
            container.total_reward = data.get('total_reward', 0.0)
            container.best_reward = data.get('best_reward', float('-inf'))
            container.best_score = data.get('best_score', 0)
            container.total_training_time = data.get('total_training_time', 0.0)
            container.created_date = data.get('created_date', datetime.now().isoformat())
            container.last_updated = data.get('last_updated', datetime.now().isoformat())
            container.model_version = data.get('model_version', '1.0')
            container.description = data.get('description', '')
            container.training_parameters = data.get('training_parameters', {})
            container.training_data = data.get('training_data', {})
            container.performance_history = data.get('performance_history', {
                'episodes': [], 'scores': [], 'rewards': [], 'timestamps': []
            })
            container.architecture = data.get('architecture', {
                'input_size': None, 'output_size': None, 'hidden_layers': None
            })
            
            print(f"✓ Model container loaded: {filepath}")
            return container
            
        except Exception as e:
            print(f"✗ Failed to load model container: {e}")
            return None
    
    def set_description(self, description):
        """Set a description for this model."""
        self.description = description
        self.last_updated = datetime.now().isoformat()
    
    def set_architecture_info(self, input_size, output_size, hidden_layers=None):
        """Set architecture information."""
        self.architecture = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layers': hidden_layers
        }
    
    def get_recent_performance(self, num_episodes=100):
        """Get recent performance statistics."""
        if not self.performance_history['episodes']:
            return None
        
        recent_scores = self.performance_history['scores'][-num_episodes:]
        recent_rewards = self.performance_history['rewards'][-num_episodes:]
        
        return {
            'avg_score': sum(recent_scores) / len(recent_scores) if recent_scores else 0,
            'avg_reward': sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0,
            'num_episodes': len(recent_scores)
        }
