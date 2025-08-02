#!/usr/bin/env python3
"""
Snake AI Project Manager

Manages complete training projects with all related files organized in project folders.
Each project contains:
- models/ (best model, checkpoints)
- logs/ (training data, CSV files)
- plots/ (training progress plots)
- config/ (training configuration, metadata)
- project.json (project metadata and settings)
"""

import os
import json
import time
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import tempfile

# Handle both relative and absolute imports
try:
    from .utils import get_user_data_dir
    from .model_container import SnakeModelContainer
except ImportError:
    from utils import get_user_data_dir
    from model_container import SnakeModelContainer


class SnakeAIProject:
    """Manages a complete Snake AI training project."""
    
    def __init__(self, project_path):
        """Initialize a project at the given path."""
        self.project_path = Path(project_path)
        self.project_name = self.project_path.name
        
        # Project structure
        self.models_dir = self.project_path / "models"
        self.logs_dir = self.project_path / "logs"
        self.plots_dir = self.project_path / "plots"
        self.config_dir = self.project_path / "config"
        self.metadata_file = self.project_path / "project.json"
        
        # Standard file paths
        self.best_model_path = self.models_dir / "best_model.pth"
        self.checkpoint_path = self.models_dir / "checkpoint.pth"
        self.training_plot_path = self.plots_dir / "training_progress.png"
        
        # Initialize metadata
        self.metadata = {
            "name": self.project_name,
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0",
            "description": "",
            "training_sessions": [],
            "best_score": 0,
            "best_reward": float('-inf'),
            "total_episodes": 0,
            "model_architecture": {
                "input_size": None,
                "output_size": None,
                "hidden_layers": None
            },
            "training_config": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "checkpoint_frequency": 500,
                "reward_weights": {
                    "wall_collision_penalty": -1.0,
                    "self_collision_penalty": -0.5,
                    "food_reward": 10.0,
                    "closer_reward": 1.0,
                    "farther_penalty": -0.5,
                    "survival_reward": -0.01
                }
            }
        }
    
    def create_project(self, description=""):
        """Create a new project with proper directory structure."""
        # Create all directories
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        # Set description
        if description:
            self.metadata["description"] = description
        
        # Save initial metadata
        self.save_metadata()
        
        print(f"✓ Created project: {self.project_name}")
        print(f"✓ Project location: {self.project_path}")
        
        return True
    
    def load_project(self):
        """Load an existing project."""
        if not self.project_path.exists():
            print(f"Project path does not exist: {self.project_path}")
            return False
        
        if not self.metadata_file.exists():
            print(f"Project metadata not found: {self.metadata_file}")
            return False
        
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded project: {self.project_name}")
            return True
        except Exception as e:
            print(f"Failed to load project metadata: {e}")
            return False
    
    def save_metadata(self):
        """Save project metadata."""
        self.metadata["last_modified"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save project metadata: {e}")
    
    def add_training_session(self, episodes_trained, best_score, best_reward, session_duration):
        """Add a training session to the project history."""
        session = {
            "timestamp": datetime.now().isoformat(),
            "episodes_trained": episodes_trained,
            "best_score": best_score,
            "best_reward": best_reward,
            "duration_seconds": session_duration,
            "total_episodes_after": self.metadata["total_episodes"] + episodes_trained
        }
        
        self.metadata["training_sessions"].append(session)
        self.metadata["total_episodes"] += episodes_trained
        
        # Update best scores if improved
        if best_score > self.metadata["best_score"]:
            self.metadata["best_score"] = best_score
        if best_reward > self.metadata["best_reward"]:
            self.metadata["best_reward"] = best_reward
        
        self.save_metadata()
    
    def get_next_log_filename(self, prefix="training_data"):
        """Get the next available log filename."""
        timestamp = int(time.time())
        session_num = len(self.metadata["training_sessions"]) + 1
        return self.logs_dir / f"{prefix}_session_{session_num}_{timestamp}.csv"
    
    def export_project_archive(self, export_path=None):
        """Export the entire project as a ZIP archive."""
        if not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.project_path.parent / f"{self.project_name}_export_{timestamp}.zip"
        
        try:
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.project_path):
                    for file in files:
                        file_path = Path(root) / file
                        arc_name = file_path.relative_to(self.project_path.parent)
                        zipf.write(file_path, arc_name)
            
            print(f"✓ Project exported to: {export_path}")
            return str(export_path)
        except Exception as e:
            print(f"Failed to export project: {e}")
            return None
    
    def import_project_archive(self, archive_path, import_location=None):
        """Import a project from a ZIP archive."""
        if not import_location:
            import_location = self.project_path.parent
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(import_location)
            
            print(f"✓ Project imported to: {import_location}")
            return True
        except Exception as e:
            print(f"Failed to import project: {e}")
            return False
    
    def get_project_summary(self):
        """Get a summary of the project."""
        if not self.metadata_file.exists():
            return "Project metadata not found"
        
        summary = []
        summary.append(f"Project: {self.metadata['name']}")
        summary.append(f"Created: {self.metadata['created'][:10]}")
        summary.append(f"Total Episodes: {self.metadata['total_episodes']}")
        summary.append(f"Best Score: {self.metadata['best_score']}")
        summary.append(f"Best Reward: {self.metadata['best_reward']:.2f}")
        summary.append(f"Training Sessions: {len(self.metadata['training_sessions'])}")
        
        if self.metadata['description']:
            summary.append(f"Description: {self.metadata['description']}")
        
        return "\n".join(summary)


class ProjectManager:
    """Manages multiple Snake AI projects."""
    
    def __init__(self):
        """Initialize the project manager."""
        self.projects_root = Path(get_user_data_dir()) / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
    
    def create_new_project(self, project_name, description=""):
        """Create a new project."""
        # Sanitize project name
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            safe_name = f"snake_ai_project_{int(time.time())}"
        
        project_path = self.projects_root / safe_name
        
        # Check if project already exists
        if project_path.exists():
            counter = 1
            while (project_path.parent / f"{safe_name}_{counter}").exists():
                counter += 1
            project_path = project_path.parent / f"{safe_name}_{counter}"
        
        project = SnakeAIProject(project_path)
        if project.create_project(description):
            return project
        return None
    
    def load_project(self, project_path):
        """Load an existing project."""
        project = SnakeAIProject(project_path)
        if project.load_project():
            return project
        return None
    
    def list_projects(self):
        """List all available projects."""
        projects = []
        for item in self.projects_root.iterdir():
            if item.is_dir():
                metadata_file = item / "project.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        projects.append({
                            'name': metadata['name'],
                            'path': str(item),
                            'created': metadata['created'],
                            'total_episodes': metadata['total_episodes'],
                            'best_score': metadata['best_score'],
                            'description': metadata.get('description', '')
                        })
                    except Exception as e:
                        print(f"Failed to read project metadata for {item.name}: {e}")
        
        return sorted(projects, key=lambda x: x['created'], reverse=True)
    
    def get_projects_directory(self):
        """Get the root directory for projects."""
        return str(self.projects_root)


def interactive_project_selection():
    """Simple project selection dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        manager = ProjectManager()
        projects_dir = manager.get_projects_directory()
        
        root = tk.Tk()
        root.withdraw()
        
        # Simple directory selection for existing projects
        selected_dir = filedialog.askdirectory(
            title="Select Snake AI Project Folder",
            initialdir=projects_dir
        )
        
        root.destroy()
        
        if selected_dir:
            # Try to load the selected project
            project = SnakeAIProject(selected_dir)
            if project.load_project():
                return project
            else:
                print(f"Invalid project folder: {selected_dir}")
                return None
        
        return None
        
    except Exception as e:
        print(f"Project selection failed: {e}")
        return None
