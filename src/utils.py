import torch
import os

def preprocess_state(state):
    # Convert the game state into a format suitable for the AI model
    return state.flatten()  # Example: flattening the state for input

def visualize_game(state, score):
    # Function to visualize the game state (optional)
    pass  # Implementation for visualization can be added here

def log_training_data(data):
    # Function to log training data for analysis
    with open('training_log.txt', 'a') as f:
        f.write(f"{data}\n")  # Append data to a log file

def save_model(model, filename):
    # Function to save the trained model
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename):
    # Function to load a trained model
    if os.path.exists(filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(filename, map_location=device))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {filename}")
        return True
    else:
        print(f"Model file {filename} not found")
        return False

def save_checkpoint(model, optimizer, episode, score, filename):
    # Function to save a complete checkpoint including optimizer state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'score': score
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    # Function to load a complete checkpoint
    if os.path.exists(filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['episode']
        score = checkpoint['score']
        print(f"Checkpoint loaded from {filename} - Episode: {episode}, Score: {score}")
        return episode, score
    else:
        print(f"Checkpoint file {filename} not found")
        return 0, 0