import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SnakeAI:
    def __init__(self, input_size, output_size, learning_rate=0.001, target_update_freq=1000):
        self.learning_rate = learning_rate  # Store learning rate as attribute
        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10_000)
        self.gamma = 0.99  # Discount factor
        self.train_step_counter = 0
        self.target_update_freq = target_update_freq

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, device, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.criterion(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
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
