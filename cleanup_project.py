#!/usr/bin/env python3
"""
Snake AI Project Cleanup Script

This script helps clean up unused files from your Snake AI project.
It categorizes files and gives you options for what to keep/remove.
"""

import os
import shutil
from pathlib import Path

class SnakeAICleanup:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        
        # Core files that should NEVER be removed
        self.core_files = {
            'src/__main__.py',
            'src/snake_game.py',
            'src/snake_ai.py', 
            'src/snake_game_renderer.py',
            'src/menu.py',
            'src/model_container.py',
            'src/project_manager.py',
            'src/training_framework.py',
            'src/strategy_analytics.py',
            'src/utils.py',
            'config.yaml',
            'requirements.txt',
            'README.md'
        }
        
        # Optional testing/development tools (can be removed if not needed)
        self.optional_tools = {
            'src/snake_gym_env.py': 'OpenAI Gym environment wrapper for RL experiments',
            'src/tensorboard_logger.py': 'TensorBoard integration for experiment tracking',
            'src/wandb_logger.py': 'Weights & Biases integration for ML experiment tracking',
            'Snake AI Strategy Testing.ipynb': 'Jupyter notebook for strategy testing'
        }
        
        # Documentation files (useful but can be archived)
        self.documentation = {
            'CONFIG.md': 'Configuration documentation',
            'CURRICULUM.md': 'Curriculum learning documentation', 
            'FIXED_DOUBLE_ITERATION_PROMPT.md': 'Bug fix documentation',
            'MOVE_COST_IMPLEMENTATION.md': 'Move cost feature documentation',
            'NO_SCORE_PENALTY.md': 'No score penalty documentation',
            'PROJECT_CONFIG_STORAGE.md': 'Project configuration documentation',
            'SECURITY_SCANNING.md': 'Security scanning documentation',
            'STRATEGY_SELECTION_GUIDE.md': 'Strategy selection guide',
            'TRAINING_FRAMEWORK.md': 'Training framework documentation'
        }
        
        # Security scanning files (optional for most users)
        self.security_files = {
            'check_dependencies.py': 'Dependency vulnerability checker',
            'check_dependencies_full.py': 'Full dependency vulnerability checker',
            'dependency_scan_config.ini': 'Security scan configuration',
            'scan_dependencies.bat': 'Windows security scan script',
            'scan_dependencies.ps1': 'PowerShell security scan script',
            'setup_scheduled_scan.bat': 'Windows scheduled scan setup',
            'requirements-security.txt': 'Security requirements file'
        }
        
        # Files that can be safely removed (old/unused)
        self.removable_files = {
            'src/view_training_plot.py': 'Replaced by integrated plotting',
            'src/custom_strategies.py': 'Replaced by strategies/ folder system',
            'test_config.py': 'Old configuration testing',
            'test_enhanced_training.py': 'Old training tests',
            'test_no_score_penalty.py': 'Old specific testing',
            'test_project_validation.py': 'Old project testing', 
            'test_reward_balance.py': 'Old reward testing',
            'test_reward_function.py': 'Old reward testing',
            'test_reward_weights.py': 'Old reward testing',
            'test_strategy_configs.py': 'Old strategy testing',
            'test_strategy_selection.py': 'Old strategy testing',
            'test_strategy_tracker.py': 'Old analytics testing',
            'demonstrate_move_cost.py': 'Old demonstration script',
            'demo_framework.py': 'Old demo script',
            'diagnose_rewards.py': 'Old diagnostic script',
            'performance_analysis.py': 'Old analysis script',
            'verify_new_settings.py': 'Old verification script',
            'inspect_project.py': 'Old inspection script'
        }
    
    def scan_project(self):
        """Scan the project and categorize all files."""
        all_files = []
        
        # Scan root directory
        for file_path in self.project_root.glob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                all_files.append(str(file_path.relative_to(self.project_root)))
        
        # Scan src directory
        src_dir = self.project_root / 'src'
        if src_dir.exists():
            for file_path in src_dir.glob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    all_files.append(str(file_path.relative_to(self.project_root)))
        
        return all_files
    
    def categorize_files(self):
        """Categorize all files in the project."""
        all_files = self.scan_project()
        
        categorized = {
            'core': [],
            'optional_tools': [],
            'documentation': [],
            'security': [],
            'removable': [],
            'unknown': []
        }
        
        for file_path in all_files:
            normalized_path = file_path.replace('\\', '/')
            
            if normalized_path in self.core_files:
                categorized['core'].append(file_path)
            elif normalized_path in self.optional_tools:
                categorized['optional_tools'].append(file_path)
            elif normalized_path in self.documentation:
                categorized['documentation'].append(file_path)
            elif normalized_path in self.security_files:
                categorized['security'].append(file_path)
            elif normalized_path in self.removable_files:
                categorized['removable'].append(file_path)
            else:
                categorized['unknown'].append(file_path)
        
        return categorized
    
    def print_analysis(self):
        """Print analysis of all files."""
        categorized = self.categorize_files()
        
        print("🐍 Snake AI Project File Analysis")
        print("=" * 50)
        
        print(f"\n✅ CORE FILES ({len(categorized['core'])} files)")
        print("These are essential and should NOT be removed:")
        for file_path in sorted(categorized['core']):
            print(f"  📁 {file_path}")
        
        print(f"\n🔧 OPTIONAL TOOLS ({len(categorized['optional_tools'])} files)")
        print("These are useful for development but not required for basic usage:")
        for file_path in sorted(categorized['optional_tools']):
            normalized = file_path.replace('\\', '/')
            description = self.optional_tools.get(normalized, 'Optional tool')
            print(f"  🛠️ {file_path} - {description}")
        
        print(f"\n📚 DOCUMENTATION ({len(categorized['documentation'])} files)")
        print("These are documentation files (can be archived if not needed):")
        for file_path in sorted(categorized['documentation']):
            normalized = file_path.replace('\\', '/')
            description = self.documentation.get(normalized, 'Documentation')
            print(f"  📄 {file_path} - {description}")
        
        print(f"\n🔒 SECURITY FILES ({len(categorized['security'])} files)")
        print("These are for security scanning (optional for most users):")
        for file_path in sorted(categorized['security']):
            normalized = file_path.replace('\\', '/')
            description = self.security_files.get(normalized, 'Security tool')
            print(f"  🔐 {file_path} - {description}")
        
        print(f"\n🗑️ REMOVABLE FILES ({len(categorized['removable'])} files)")
        print("These can be safely removed (old/unused):")
        for file_path in sorted(categorized['removable']):
            normalized = file_path.replace('\\', '/')
            description = self.removable_files.get(normalized, 'Old/unused file')
            print(f"  ❌ {file_path} - {description}")
        
        if categorized['unknown']:
            print(f"\n❓ UNKNOWN FILES ({len(categorized['unknown'])} files)")
            print("These files are not categorized:")
            for file_path in sorted(categorized['unknown']):
                print(f"  ❓ {file_path}")
        
        return categorized
    
    def remove_files(self, file_list, backup=True):
        """Remove specified files with optional backup."""
        if backup:
            backup_dir = self.project_root / 'removed_files_backup'
            backup_dir.mkdir(exist_ok=True)
        
        removed = []
        errors = []
        
        for file_path in file_list:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                errors.append(f"File not found: {file_path}")
                continue
            
            try:
                if backup:
                    backup_path = backup_dir / file_path.replace('/', '_').replace('\\', '_')
                    shutil.copy2(full_path, backup_path)
                
                full_path.unlink()
                removed.append(file_path)
                print(f"✅ Removed: {file_path}")
                
            except Exception as e:
                errors.append(f"Error removing {file_path}: {e}")
        
        return removed, errors
    
    def interactive_cleanup(self):
        """Interactive cleanup with user choices."""
        print("🧹 Interactive Snake AI Project Cleanup")
        print("=" * 40)
        
        categorized = self.print_analysis()
        
        print("\n🎯 Recommended Actions:")
        print("1. Remove old/unused files (safe)")
        print("2. Optionally remove security files (if not needed)")
        print("3. Archive documentation (move to docs/ folder)")
        print("4. Keep optional tools (or remove if not doing RL research)")
        
        # Auto-remove obviously unused files
        if categorized['removable']:
            response = input(f"\n🗑️ Remove {len(categorized['removable'])} old/unused files? (y/n): ").lower()
            if response == 'y':
                removed, errors = self.remove_files(categorized['removable'])
                print(f"✅ Removed {len(removed)} files")
                if errors:
                    print("❌ Errors:")
                    for error in errors:
                        print(f"  {error}")
        
        # Optional: Remove security files
        if categorized['security']:
            response = input(f"\n🔒 Remove {len(categorized['security'])} security scanning files? (y/n): ").lower()
            if response == 'y':
                removed, errors = self.remove_files(categorized['security'])
                print(f"✅ Removed {len(removed)} security files")
        
        # Optional: Archive documentation
        if categorized['documentation']:
            response = input(f"\n📚 Move {len(categorized['documentation'])} documentation files to docs/ folder? (y/n): ").lower()
            if response == 'y':
                docs_dir = self.project_root / 'docs'
                docs_dir.mkdir(exist_ok=True)
                
                for file_path in categorized['documentation']:
                    src_path = self.project_root / file_path
                    dst_path = docs_dir / Path(file_path).name
                    
                    if src_path.exists():
                        shutil.move(str(src_path), str(dst_path))
                        print(f"📁 Moved: {file_path} → docs/{Path(file_path).name}")
        
        print("\n🎉 Cleanup completed!")
        print("\nYour project now contains only:")
        print("✅ Core Snake AI application files")
        print("✅ Strategy framework")
        print("✅ Configuration files")
        print("✅ Essential utilities")
        
        if any(categorized['optional_tools']):
            print("🔧 Optional development tools (kept)")


def main():
    """Main cleanup function."""
    project_root = Path(__file__).parent
    cleanup = SnakeAICleanup(project_root)
    
    # Run interactive cleanup
    cleanup.interactive_cleanup()


if __name__ == "__main__":
    main()
