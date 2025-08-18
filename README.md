# Snake AI Project

This project implements a Snake game that utilizes artificial intelligence to make decisions based on the game state. The AI is built using PyTorch and uses a project-based system to organize training sessions, models, and data.

## Key Features

- **Project-Based Organization**: All training data, models, checkpoints, and logs are organized into self-contained project folders
- **Menu-Driven Interface**: Easy-to-use pygame menu system for creating and managing projects
- **Automatic Data Management**: Training plots, CSV logs, and model checkpoints are automatically saved and organized
- **Roaming Data Storage**: Projects are stored in the user's roaming data directory for proper OS integration
- **Export/Import**: Projects can be exported as ZIP archives for backup or sharing
- **Security Scanning**: Automated dependency and security vulnerability scanning

## Project Structure

```
snake-ai/
├── src/                      # Source code
│   ├── snake_ai/            # Main package
│   │   ├── core/            # Core game logic
│   │   ├── strategies/      # AI strategies
│   │   ├── training/        # Training framework
│   │   └── utils/           # Utilities
│   └── __main__.py          # Application entry point
├── tests/                   # Test suite
├── tools/                   # Development tools
│   ├── security/           # Security scanning tools
│   └── README.md           # Tools documentation
├── docs/                   # Documentation
├── config/                 # Configuration files
│   ├── strategies/         # Strategy configurations
│   └── security_scan.ini   # Security scan settings
├── examples/               # Example code and strategies
├── requirements/           # Requirements files
│   ├── base.txt           # Core dependencies
│   ├── dev.txt            # Development dependencies
│   └── security.txt       # Security tools
├── logs/                  # Log files (auto-generated)
├── pyproject.toml         # Modern Python packaging
├── requirements.txt       # Legacy requirements (for compatibility)
└── README.md              # This documentation
```

## Project Data Structure

When you create a project, it automatically creates an organized structure:

```
Project_Name/
├── models/
│   ├── best_model.pth        # Best performing model
│   └── checkpoint.pth        # Latest training checkpoint
├── logs/
│   └── training_data_*.csv   # Training session logs
├── plots/
│   └── training_progress.png # Training visualization
├── config/
│   └── (future: training configurations)
└── project.json              # Project metadata and history
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd snake-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

To run the Snake AI application:
```
python -m src
```

### Menu System

The application features a streamlined menu-driven interface:

1. **Create New Project**: Choose location and name via file dialog
2. **Load Project**: Select existing project folder
3. **AI Demo**: Watch the AI play (requires a trained project)
4. **Manual Mode**: Play the game yourself
5. **Training Mode**: Train the AI (requires a project)

### Training Workflow

1. **Create Project**: Choose "Create New Project" and select where to save it
2. **Start Training**: Select "Training Mode" and specify the number of episodes
3. **Monitor Progress**: Training data is automatically saved and organized
4. **Review Results**: Training plots and CSV logs are saved in the project

### Project Creation

Creating a new project is simple:
- Select "Create New Project" from the menu
- Use the file dialog to choose where to save your project
- Type the project name (e.g., "Snake_AI_Training")
- The system automatically creates the complete project structure

### Loading Projects

Loading an existing project is equally simple:
- Select "Load Project" from the menu
- Browse to and select your project folder
- The system automatically loads all models and training history

### Data Storage

Projects are stored in your system's roaming data directory:
- **Windows**: `%APPDATA%\SnakeAI\projects\`
- **macOS**: `~/Library/Application Support/SnakeAI/projects/`
- **Linux**: `~/.config/SnakeAI/projects/`

## AI Architecture

- **Deep Q-Network (DQN)** with experience replay
- **Input**: Game state represented as a grid
- **Output**: Action probabilities (straight, left, right)
- **Reward System**: Optimized for food collection and survival

## Training Features

- **Automatic Checkpointing**: Models are saved periodically during training
- **Best Model Tracking**: The best performing model is automatically saved
- **Training Logs**: Detailed CSV logs with episode data, rewards, and statistics
- **Progress Visualization**: Automatic generation of training progress plots
- **Session History**: Each project tracks all training sessions with metadata

## Security and Maintenance

### Security Scanning

The project includes automated security scanning tools:

```bash
# Quick security scan (dependencies only)
python tools/security_tools.py --quick

# Full security scan (includes static analysis)
python tools/security_tools.py --full

# Install security tools
python tools/security_tools.py --install
```

Security scans check for:
- Known vulnerabilities in dependencies
- Outdated packages with security updates
- Static code security issues
- Dependency tree analysis

Results are saved to `logs/security_scan_*.txt` for review.

### Installation Options

#### Standard Installation
```bash
# Core dependencies only
pip install -r requirements/base.txt

# Development dependencies
pip install -r requirements/dev.txt

# Security tools
pip install -r requirements/security.txt
```

#### Modern Installation (Recommended)
```bash
# Core installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With security tools
pip install -e ".[security]"

# Everything
pip install -e ".[dev,security,docs]"
```

## Export and Backup

Projects can be exported as ZIP archives containing all models, data, and metadata. This makes it easy to:
- Share trained models with others
- Backup your training progress
- Transfer projects between computers

## Overview

This project demonstrates reinforcement learning in action with a complete project management system. The AI learns to play Snake through trial and error, with all training data automatically organized and preserved for analysis and future use.