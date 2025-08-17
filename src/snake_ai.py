import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Deeper 4-layer network architecture optimized for feature learning
        # Architecture: input -> 64 -> 128 -> 64 -> output
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
        # Dropout for regularization - prevent overfitting
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)  # Higher dropout in middle layer
        self.dropout3 = nn.Dropout(0.2)
        
        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Ensure input is properly shaped
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Layer 1: Input -> 64 features
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Layer 2: 64 -> 128 features (expansion layer)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Layer 3: 128 -> 64 features (compression layer)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        
        # Output layer: 64 -> 3 actions (no activation - raw Q-values)
        x = self.fc4(x)
        
        return x

class SnakeAI:
    def __init__(self, input_size, output_size, learning_rate=0.0003, target_update_freq=1000):
        self.learning_rate = learning_rate  # Optimized learning rate for deeper network
        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Adam optimizer with weight decay for better generalization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=50_000)  # Increased memory for deeper network stability
        self.gamma = 0.99  # Discount factor
        self.train_step_counter = 0
        self.target_update_freq = target_update_freq

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, device, batch_size=256):  # Larger batch size for deeper network stability
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Set model to training mode for dropout
        self.model.train()
        current_q = self.model(states).gather(1, actions)
        
        # Set target model to eval mode
        self.target_model.eval()
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability with deeper network
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon, device):
        if np.random.rand() <= epsilon:
            return np.random.choice([0, 1, 2])  # Random action
        
        # Use the model to predict the best action
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():  # Disable gradients for inference
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        
        return action
