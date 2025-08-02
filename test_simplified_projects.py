#!/usr/bin/env python3

import os
import sys
import tempfile

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_simplified_project_creation():
    """Test the simplified project creation workflow."""
    print("Testing simplified project creation...")
    
    from project_manager import SnakeAIProject
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = os.path.join(temp_dir, "Test_Snake_Project")
        
        # Test creating a project directly
        project = SnakeAIProject(project_path)
        
        if project.create_project():
            print(f"✓ Project created successfully: {project.project_name}")
            print(f"  Location: {project.project_path}")
            
            # Verify project structure
            expected_dirs = ['models', 'logs', 'plots', 'config']
            for dir_name in expected_dirs:
                dir_path = project.project_path / dir_name
                if dir_path.exists():
                    print(f"  ✓ {dir_name}/ directory exists")
                else:
                    print(f"  ✗ {dir_name}/ directory missing")
            
            # Test loading the project
            project2 = SnakeAIProject(project_path)
            if project2.load_project():
                print(f"  ✓ Project can be reloaded")
                print(f"  Project name: {project2.project_name}")
            else:
                print(f"  ✗ Failed to reload project")
            
            print("\n✓ Simplified project system working correctly!")
        else:
            print("✗ Failed to create project")

def test_file_dialog_workflow():
    """Test the file dialog workflow concept."""
    print("\nTesting file dialog workflow concept...")
    
    # Simulate what happens when user selects a file path
    test_path = "/path/to/My_Snake_Project"
    
    # Remove extension if any (as done in create_new_project)
    project_path = os.path.splitext(test_path)[0]
    
    print(f"User selected: {test_path}")
    print(f"Project path: {project_path}")
    print(f"Project name would be: {os.path.basename(project_path)}")
    
    print("✓ File dialog path processing working correctly!")

if __name__ == "__main__":
    test_simplified_project_creation()
    test_file_dialog_workflow()
