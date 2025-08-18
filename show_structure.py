#!/usr/bin/env python3
"""
Script to show the organized project structure
"""

from pathlib import Path

def show_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Display directory tree structure"""
    if current_depth >= max_depth:
        return
    
    directory = Path(directory)
    items = []
    
    try:
        # Get all items and sort them (directories first, then files)
        all_items = list(directory.iterdir())
        directories = [item for item in all_items if item.is_dir() and not item.name.startswith('.')]
        files = [item for item in all_items if item.is_file() and not item.name.startswith('.')]
        items = sorted(directories) + sorted(files)
    except PermissionError:
        return
    
    for i, item in enumerate(items):
        # Skip certain directories and files
        skip_items = {
            '__pycache__', '.git', '.vscode', '.idea', 'node_modules', 
            '.pytest_cache', '.coverage', 'htmlcov', 'removed_files_backup'
        }
        
        if item.name in skip_items:
            continue
            
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir():
            extension = "    " if is_last else "│   "
            show_tree(item, prefix + extension, max_depth, current_depth + 1)

def main():
    """Show the organized project structure"""
    print("🐍 Snake AI Project - Organized Structure")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    print(f"📁 {project_root.name}/")
    show_tree(project_root)
    
    print("\n📋 Structure Summary:")
    print("├── src/           - Source code and main package")
    print("├── tests/         - Test suite") 
    print("├── tools/         - Development and security tools")
    print("├── docs/          - Documentation")
    print("├── config/        - Configuration files")
    print("├── examples/      - Example code and strategies")
    print("├── requirements/  - Organized dependency files")
    print("├── logs/          - Generated log files (git-ignored)")
    print("└── pyproject.toml - Modern Python packaging")
    
    print("\n🔧 Quick Commands:")
    print("• Run application:      python -m src")
    print("• Install dependencies: pip install -e .")
    print("• Security scan:        python tools/security_tools.py")
    print("• Run tests:            pytest")
    print("• Format code:          black src/ tests/")
    
    print("\n📦 Installation Options:")
    print("• Core only:      pip install -e .")
    print("• With dev tools: pip install -e '.[dev]'")
    print("• With security:  pip install -e '.[security]'")
    print("• Everything:     pip install -e '.[dev,security,docs]'")

if __name__ == "__main__":
    main()
