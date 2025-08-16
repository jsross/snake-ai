import torch
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import glob
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .model_container import SnakeModelContainer
except ImportError:
    from model_container import SnakeModelContainer

def get_user_data_dir():
    """Get the appropriate user data directory for the current OS."""
    if os.name == 'nt':  # Windows
        app_data = os.environ.get('APPDATA')
        if app_data:
            return os.path.join(app_data, 'SnakeAI')
    
    # Fallback for other systems or if APPDATA not found
    home = os.path.expanduser('~')
    if os.name == 'nt':  # Windows fallback
        return os.path.join(home, 'AppData', 'Roaming', 'SnakeAI')
    elif os.name == 'posix':  # Linux/Mac
        # Use XDG Base Directory spec for Linux, or ~/Library/Application Support for Mac
        if 'darwin' in os.sys.platform.lower():  # Mac
            return os.path.join(home, 'Library', 'Application Support', 'SnakeAI')
        else:  # Linux
            return os.path.join(home, '.config', 'SnakeAI')
    
    # Ultimate fallback
    return os.path.join(home, '.snake_ai')

def save_unified_model(model, optimizer, filepath, episode=0, score=0, total_reward=0.0, training_data=None, description=""):
    """Save model using the unified container system."""
    container = SnakeModelContainer()
    
    # Update container with training information
    container.model_state_dict = model.state_dict()
    container.optimizer_state_dict = optimizer.state_dict()
    container.episode = episode
    container.score = score
    container.total_reward = total_reward
    
    if training_data:
        container.training_data.update(training_data)
    
    if description:
        container.set_description(description)
    
    return container.save(filepath)

def load_unified_model(filepath, model, optimizer=None, target_model=None, device=None):
    """Load model using the unified container system."""
    container = SnakeModelContainer.load(filepath, device)
    if container:
        success = container.load_into_model(model, optimizer, target_model, device)
        return success, container
    return False, None

def load_model(model, filename, optimizer=None, target_model=None):
    """Load a model - handles legacy formats and new unified containers."""
    if not os.path.exists(filename):
        print(f"Model file {filename} not found")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try to load as unified container first
        success, container = load_unified_model(filename, model, optimizer, target_model, device)
        if success:
            return True
        
        # Fall back to legacy loading
        print("Trying legacy model format...")
        data = torch.load(filename, map_location=device, weights_only=False)  # Allow all content for legacy files
        
        # Check if this is a checkpoint file or a regular model file
        if isinstance(data, dict) and 'model_state_dict' in data:
            # This is a legacy checkpoint file
            model.load_state_dict(data['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in data:
                optimizer.load_state_dict(data['optimizer_state_dict'])
            print(f"Model loaded from legacy checkpoint: {filename}")
            if 'episode' in data and 'score' in data:
                print(f"  Checkpoint info - Episode: {data['episode']}, Score: {data['score']}")
        else:
            # This is a legacy model state dict
            model.load_state_dict(data)
            print(f"Model loaded from legacy format: {filename}")
        
        model.eval()  # Set the model to evaluation mode
        return True
        
    except Exception as e:
        print(f"Failed to load model from {filename}: {e}")
        return False

def load_model_with_tracking(model, filename):
    """Load a model and save its path as the last loaded model."""
    success = load_model(model, filename)
    if success:
        # Save this as the last loaded model
        save_last_model_path(filename)
    return success

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
        checkpoint = torch.load(filename, map_location=device, weights_only=False)  # Allow all content for checkpoints
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['episode']
        score = checkpoint['score']
        print(f"Checkpoint loaded from {filename} - Episode: {episode}, Score: {score}")
        return episode, score
    else:
        print(f"Checkpoint file {filename} not found")
        return 0, 0

# === MODEL FILE LOADER UTILITIES ===

def save_last_model_path(model_path, models_dir="models"):
    """Save the path of the last loaded model to user data directory."""
    try:
        user_data_dir = get_user_data_dir()
        os.makedirs(user_data_dir, exist_ok=True)
        last_model_file = os.path.join(user_data_dir, "last_model.txt")
        
        # Convert to absolute path for better reliability
        abs_model_path = os.path.abspath(model_path)
        
        with open(last_model_file, 'w') as f:
            f.write(abs_model_path)
        
        print(f"Last model path saved to: {last_model_file}")
    except Exception as e:
        print(f"Failed to save last model path: {e}")

def load_last_model_path(models_dir="models"):
    """Load the path of the last loaded model from user data directory."""
    try:
        user_data_dir = get_user_data_dir()
        last_model_file = os.path.join(user_data_dir, "last_model.txt")
        
        if os.path.exists(last_model_file):
            with open(last_model_file, 'r') as f:
                path = f.read().strip()
                if path and os.path.exists(path):
                    print(f"Found last model path: {path}")
                    return path
                else:
                    print(f"Last model path exists but file not found: {path}")
        else:
            print(f"No last model file found at: {last_model_file}")
    except Exception as e:
        print(f"Failed to load last model path: {e}")
    return None

def get_last_model_info():
    """Get information about the last model configuration."""
    try:
        user_data_dir = get_user_data_dir()
        last_model_file = os.path.join(user_data_dir, "last_model.txt")
        
        info = {
            'user_data_dir': user_data_dir,
            'last_model_file': last_model_file,
            'exists': os.path.exists(last_model_file),
            'last_model_path': None
        }
        
        if info['exists']:
            try:
                with open(last_model_file, 'r') as f:
                    info['last_model_path'] = f.read().strip()
            except Exception as e:
                info['error'] = str(e)
        
        return info
    except Exception as e:
        return {'error': str(e)}

def list_available_models(models_dir="models"):
    """
    List all available model files in the models directory.
    Returns a list of model file paths.
    """
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found")
        return []
    
    # Look for PyTorch model files
    model_extensions = ["*.pth", "*.pt", "*.pkl"]
    model_files = []
    
    for extension in model_extensions:
        pattern = os.path.join(models_dir, extension)
        model_files.extend(glob.glob(pattern))
    
    # Convert to relative paths for cleaner display
    model_files = [os.path.relpath(f) for f in model_files]
    model_files.sort()  # Sort alphabetically
    
    return model_files

def select_model_file_gui(models_dir="models", title="Select Model File"):
    """
    Open a GUI file picker to select a model file.
    Returns the selected file path or None if cancelled.
    """
    try:
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Set initial directory
        initial_dir = os.path.abspath(models_dir) if os.path.exists(models_dir) else os.getcwd()
        
        # File types for the dialog
        filetypes = [
            ("PyTorch Models", "*.pth *.pt"),
            ("Pickle Files", "*.pkl"),
            ("All Files", "*.*")
        ]
        
        # Open file dialog
        filename = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        root.destroy()  # Clean up
        
        return filename if filename else None
        
    except Exception as e:
        print(f"GUI file picker failed: {e}")
        return None

def select_model_file_console(models_dir="models"):
    """
    Console-based model file selection.
    Returns the selected file path or None if cancelled.
    """
    available_models = list_available_models(models_dir)
    
    if not available_models:
        print("No model files found in the models directory")
        return None
    
    print("\nAvailable model files:")
    print("-" * 40)
    for i, model_file in enumerate(available_models, 1):
        # Get file info
        file_path = model_file
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            mod_time = os.path.getmtime(file_path)
            import time
            mod_time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))
            print(f"{i:2}. {os.path.basename(model_file)}")
            print(f"     Size: {file_size_mb:.1f} MB, Modified: {mod_time_str}")
        else:
            print(f"{i:2}. {os.path.basename(model_file)}")
    
    print(f"{len(available_models) + 1:2}. Browse for other file...")
    print(f"{len(available_models) + 2:2}. Cancel")
    print("-" * 40)
    
    while True:
        try:
            choice = input(f"Select model (1-{len(available_models) + 2}): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(available_models):
                selected_file = available_models[choice_num - 1]
                print(f"Selected: {selected_file}")
                return selected_file
            elif choice_num == len(available_models) + 1:
                # Browse for other file
                return select_model_file_gui(models_dir, "Browse for Model File")
            elif choice_num == len(available_models) + 2:
                # Cancel
                print("Selection cancelled")
                return None
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(available_models) + 2}")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None

def load_model_gui_simple(model, models_dir="models"):
    """
    Simplified GUI-based model loading with minimal dialogs.
    
    Args:
        model: The PyTorch model to load weights into
        models_dir: Directory containing model files (used as initial directory)
    
    Returns:
        bool: True if model was loaded successfully, False otherwise
    """
    try:
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Set initial directory
        initial_dir = os.path.abspath(models_dir) if os.path.exists(models_dir) else os.getcwd()
        
        # File types for the dialog
        filetypes = [
            ("PyTorch Models", "*.pth *.pt"),
            ("Pickle Files", "*.pkl"),
            ("All Files", "*.*")
        ]
        
        # Open file dialog directly (no info dialog first)
        selected_file = filedialog.askopenfilename(
            title="Select Model File to Load",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        if not selected_file:
            # User cancelled - no dialog needed
            root.destroy()
            return False
        
        # Try to load the model
        success = load_model_with_tracking(model, selected_file)
        
        # Only show success dialog, no error dialog (errors go to console)
        if success:
            messagebox.showinfo("Model Loaded", f"✓ Model loaded successfully!\n\n{os.path.basename(selected_file)}")
        
        root.destroy()
        return success
        
    except Exception as e:
        try:
            root.destroy()
        except:
            pass
        print(f"GUI model loader failed: {e}")
        return False
    """
    Pure GUI-based model loading with file selection.
    Shows a file picker dialog and loads the selected model.
    
    Args:
        model: The PyTorch model to load weights into
        models_dir: Directory containing model files (used as initial directory)
    
    Returns:
        bool: True if model was loaded successfully, False otherwise
    """
    try:
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Set initial directory
        initial_dir = os.path.abspath(models_dir) if os.path.exists(models_dir) else os.getcwd()
        
        # File types for the dialog
        filetypes = [
            ("PyTorch Models", "*.pth *.pt"),
            ("Pickle Files", "*.pkl"),
            ("All Files", "*.*")
        ]
        
        # Show a message box first to explain what's happening
        messagebox.showinfo(
            "Load Model", 
            "Select a model file to load.\n\nSupported formats:\n• .pth (PyTorch models)\n• .pt (PyTorch models)\n• .pkl (Pickle files)"
        )
        
        # Open file dialog
        selected_file = filedialog.askopenfilename(
            title="Select Model File to Load",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        if not selected_file:
            # User cancelled
            messagebox.showinfo("Load Model", "No model file selected.")
            root.destroy()
            return False
        
        # Get model info to show to user
        model_info = get_model_info(selected_file)
        
        # Show model information and confirm loading
        if model_info:
            info_text = f"Model Information:\n\n"
            info_text += f"File: {os.path.basename(selected_file)}\n"
            info_text += f"Type: {model_info.get('type', 'Unknown').title()}\n"
            info_text += f"Size: {model_info.get('file_size_mb', 0):.1f} MB\n"
            
            if 'num_parameters' in model_info:
                info_text += f"Parameters: {model_info['num_parameters']:,}\n"
                
            if model_info.get('type') == 'checkpoint':
                info_text += f"Episode: {model_info.get('episode', 'Unknown')}\n"
                info_text += f"Score: {model_info.get('score', 'Unknown')}\n"
            
            info_text += f"\nDo you want to load this model?"
            
            # Ask for confirmation
            confirm = messagebox.askyesno("Confirm Model Loading", info_text)
            if not confirm:
                messagebox.showinfo("Load Model", "Model loading cancelled.")
                root.destroy()
                return False
        
        # Try to load the model
        success = load_model(model, selected_file)
        
        if success:
            messagebox.showinfo("Success", f"Model loaded successfully!\n\nFile: {os.path.basename(selected_file)}")
        else:
            messagebox.showerror("Error", f"Failed to load model from:\n{selected_file}")
        
        root.destroy()
        return success
        
    except Exception as e:
        try:
            messagebox.showerror("Error", f"An error occurred while loading the model:\n\n{str(e)}")
            root.destroy()
        except:
            pass
        print(f"GUI model loader failed: {e}")
        return False

def load_model_interactive(model, models_dir="models", use_gui=True):
    """
    Interactive model loading with file selection.
    Now defaults to GUI-only mode.
    
    Args:
        model: The PyTorch model to load weights into
        models_dir: Directory containing model files
        use_gui: Whether to use GUI file picker (True) or console selection (False)
    
    Returns:
        bool: True if model was loaded successfully, False otherwise
    """
    if use_gui:
        # Use simplified GUI mode
        return load_model_gui_simple(model, models_dir)
    else:
        # Fallback to console mode
        print("=== Interactive Model Loader ===")
        selected_file = select_model_file_console(models_dir)
        
        if not selected_file:
            print("No model file selected")
            return False
        
        # Validate file exists
        if not os.path.exists(selected_file):
            print(f"Selected file does not exist: {selected_file}")
            return False
        
        # Try to load the model
        print(f"Loading model from: {selected_file}")
        success = load_model(model, selected_file)
        
        if success:
            print("✓ Model loaded successfully!")
        else:
            print("✗ Failed to load model")
        
        return success

def interactive_model_load_with_path_setup(model, current_models_dir="models"):
    """
    Interactive model loading that also sets up new save paths.
    Uses roaming data directory as default save location with fallback to user folder.
    
    Returns:
        dict: {
            'success': bool,
            'model_path': new path for best model saves,
            'checkpoint_path': new path for checkpoints,
            'base_dir': new base directory
        }
    """
    try:
        # Determine initial directory for file dialog
        user_data_dir = get_user_data_dir()
        models_data_dir = os.path.join(user_data_dir, "models")
        
        # Create the models directory in user data if it doesn't exist
        os.makedirs(models_data_dir, exist_ok=True)
        
        # Use user data models directory if it exists and has models, otherwise use current_models_dir
        if os.path.exists(models_data_dir) and any(f.endswith('.pth') for f in os.listdir(models_data_dir)):
            initial_dir = models_data_dir
        elif os.path.exists(current_models_dir):
            initial_dir = current_models_dir
        else:
            initial_dir = models_data_dir  # Default to user data directory
        
        root = tk.Tk()
        root.withdraw()
        
        model_file = filedialog.askopenfilename(
            title="Select Model to Load",
            filetypes=[
                ("PyTorch Models", "*.pth"),
                ("All Files", "*.*")
            ],
            initialdir=initial_dir
        )
        
        if not model_file:
            root.destroy()
            return {'success': False}
        
        # Try to load the selected model
        success = load_model(model, model_file)
        if not success:
            messagebox.showerror("Error", f"Failed to load model from:\n{model_file}")
            root.destroy()
            return {'success': False}
        
        # Determine save directory based on roaming data with fallback
        # Priority: 1) User data models dir, 2) Same dir as loaded model, 3) User home fallback
        if os.path.exists(models_data_dir):
            save_dir = models_data_dir
        else:
            # Fallback to same directory as loaded model, or user home
            loaded_model_dir = os.path.dirname(model_file)
            if os.access(loaded_model_dir, os.W_OK):
                save_dir = loaded_model_dir
            else:
                # Ultimate fallback to user home directory
                save_dir = os.path.join(os.path.expanduser('~'), 'SnakeAI', 'models')
                os.makedirs(save_dir, exist_ok=True)
        
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        
        result = {
            'success': True,
            'model_path': os.path.join(save_dir, f"{model_name}_best.pth"),
            'checkpoint_path': os.path.join(save_dir, f"{model_name}_checkpoint.pth"),
            'base_dir': save_dir
        }
        
        print(f"✓ Model loaded from: {model_file}")
        print(f"✓ Future saves will go to: {save_dir}")
        save_last_model_path(model_file)  # Remember this choice
        
        root.destroy()
        return result
        
    except Exception as e:
        print(f"Interactive model loading failed: {e}")
        return {'success': False}

def initialize_model_with_dialogs(model, default_models_dir="models"):
    """
    Initialize model with full user control over file locations.
    Gives options to create new model at chosen location or load existing model.
    
    Returns:
        dict: {
            'status': 'new', 'loaded', or 'cancelled',
            'model_path': path for saving best models,
            'checkpoint_path': path for saving checkpoints,
            'base_dir': directory for training outputs
        }
    """
    print("=== Model Initialization ===")
    
    try:
        root = tk.Tk()
        root.withdraw()
        
        # Ask user what they want to do
        choice = messagebox.askyesnocancel(
            "Model Setup",
            "How would you like to set up your model?\n\n"
            "• Yes: Load an existing model\n"
            "• No: Create a new model (choose save location)\n"
            "• Cancel: Exit application"
        )
        
        if choice is None:  # Cancel
            root.destroy()
            return {'status': 'cancelled'}
        
        elif choice:  # Yes - Load existing model
            model_file = filedialog.askopenfilename(
                title="Select Model to Load",
                filetypes=[
                    ("PyTorch Models", "*.pth"),
                    ("All Files", "*.*")
                ],
                initialdir=default_models_dir if os.path.exists(default_models_dir) else os.getcwd()
            )
            
            if not model_file:
                root.destroy()
                return {'status': 'cancelled'}
            
            # Try to load the selected model
            success = load_model(model, model_file)
            if not success:
                messagebox.showerror("Error", f"Failed to load model from:\n{model_file}")
                root.destroy()
                return {'status': 'cancelled'}
            
            # Use the directory of the loaded model for future saves
            model_dir = os.path.dirname(model_file)
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            
            result = {
                'status': 'loaded',
                'model_path': os.path.join(model_dir, f"{model_name}_best.pth"),
                'checkpoint_path': os.path.join(model_dir, f"{model_name}_checkpoint.pth"),
                'base_dir': model_dir
            }
            
            print(f"✓ Model loaded from: {model_file}")
            print(f"✓ Future saves will go to: {model_dir}")
            save_last_model_path(model_file)  # Remember this choice
            
        else:  # No - Create new model
            # Ask where to save the new model
            save_file = filedialog.asksaveasfilename(
                title="Choose Location for New Model",
                defaultextension=".pth",
                filetypes=[
                    ("PyTorch Models", "*.pth"),
                    ("All Files", "*.*")
                ],
                initialdir=default_models_dir if os.path.exists(default_models_dir) else os.getcwd(),
                initialfile="snake_model.pth"
            )
            
            if not save_file:
                root.destroy()
                return {'status': 'cancelled'}
            
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_file)
            os.makedirs(save_dir, exist_ok=True)
            
            model_name = os.path.splitext(os.path.basename(save_file))[0]
            
            result = {
                'status': 'new',
                'model_path': save_file,
                'checkpoint_path': os.path.join(save_dir, f"{model_name}_checkpoint.pth"),
                'base_dir': save_dir
            }
            
            print(f"✓ New model will be saved to: {save_file}")
            print(f"✓ Training outputs will go to: {save_dir}")
        
        root.destroy()
        return result
        
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return {'status': 'cancelled'}

def auto_load_last_model(model, models_dir="models"):
    """
    Automatically load the last model that was manually loaded.
    If no last model exists, offer options to user.
    
    Returns:
        str: Status - "loaded", "selected", "new", or "cancelled"
    """
    print("=== Model Initialization ===")
    
    # Try to load the last model
    last_model_path = load_last_model_path(models_dir)
    
    if last_model_path and os.path.exists(last_model_path):
        print(f"Found last used model: {os.path.basename(last_model_path)}")
        if load_model(model, last_model_path):
            print("✓ Last model loaded successfully!")
            return "loaded"
        else:
            print("✗ Failed to load last model")
    
    # No last model or failed to load - check for available models
    available_models = list_available_models(models_dir)
    
    if not available_models:
        print("No model files found. Starting with a new model.")
        return "new"
    
    # Ask user what they want to do
    try:
        root = tk.Tk()
        root.withdraw()
        
        choice = messagebox.askyesnocancel(
            "Model Setup",
            f"No previous model found.\n\n"
            f"Found {len(available_models)} model file(s) in {models_dir}.\n\n"
            f"Would you like to:\n"
            f"• Yes: Select a model to load\n"
            f"• No: Start with a new model\n"
            f"• Cancel: Exit application"
        )
        
        root.destroy()
        
        if choice is None:  # Cancel
            return "cancelled"
        elif choice:  # Yes - select model
            if load_model_gui_simple(model, models_dir):
                return "selected"
            else:
                print("No model selected. Starting with a new model.")
                return "new"
        else:  # No - new model
            print("Starting with a new model.")
            return "new"
            
    except Exception as e:
        print(f"GUI initialization failed: {e}")
        print("Starting with a new model.")
        return "new"

def get_model_info(filename):
    """
    Get information about a model file without loading it.
    Returns a dictionary with file information.
    """
    if not os.path.exists(filename):
        return None
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load as regular model first
        try:
            state_dict = torch.load(filename, map_location=device, weights_only=False)  # Allow all content
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # This is a checkpoint file
                info = {
                    'type': 'checkpoint',
                    'episode': state_dict.get('episode', 'Unknown'),
                    'score': state_dict.get('score', 'Unknown'),
                    'file_size_mb': os.path.getsize(filename) / (1024 * 1024),
                    'num_parameters': sum(p.numel() for p in state_dict['model_state_dict'].values())
                }
            else:
                # This is a regular model state dict
                info = {
                    'type': 'model',
                    'file_size_mb': os.path.getsize(filename) / (1024 * 1024),
                    'num_parameters': sum(p.numel() for p in state_dict.values())
                }
            
            return info
            
        except Exception as e:
            return {
                'type': 'unknown',
                'error': str(e),
                'file_size_mb': os.path.getsize(filename) / (1024 * 1024)
            }
            
    except Exception as e:
        return {
            'type': 'error',
            'error': str(e),
            'file_size_mb': os.path.getsize(filename) / (1024 * 1024) if os.path.exists(filename) else 0
        }