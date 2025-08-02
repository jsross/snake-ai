#!/usr/bin/env python3

import os
import sys

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_project_system():
    """Test the project-based system."""
    print("Testing project-based system...")
    
    from project_manager import ProjectManager, SnakeAIProject
    
    # Test project manager
    manager = ProjectManager()
    print(f"Projects directory: {manager.get_projects_directory()}")
    
    # List existing projects
    projects = manager.list_projects()
    print(f"Found {len(projects)} existing projects")
    
    # Test creating a new project
    test_project = manager.create_new_project("Test Project", "Test project for validation")
    if test_project:
        print(f"✓ Created test project: {test_project.project_name}")
        print(f"  Location: {test_project.project_path}")
        
        # Test project structure
        expected_dirs = ['models', 'logs', 'plots', 'config']
        for dir_name in expected_dirs:
            dir_path = test_project.project_path / dir_name
            if dir_path.exists():
                print(f"  ✓ {dir_name}/ directory created")
            else:
                print(f"  ✗ {dir_name}/ directory missing")
        
        # Test metadata
        if test_project.metadata_file.exists():
            print(f"  ✓ project.json metadata file created")
        else:
            print(f"  ✗ project.json metadata file missing")
        
        # Test project summary
        print("\nProject Summary:")
        print(test_project.get_project_summary())
        
        # Test file paths
        print(f"\nProject file paths:")
        print(f"  Best model: {test_project.best_model_path}")
        print(f"  Checkpoint: {test_project.checkpoint_path}")
        print(f"  Training plot: {test_project.training_plot_path}")
        
        # Clean up test project
        import shutil
        shutil.rmtree(test_project.project_path)
        print(f"\n✓ Test project cleaned up")
    else:
        print("✗ Failed to create test project")
    
    print("\n✓ Project system test completed!")

if __name__ == "__main__":
    test_project_system()
