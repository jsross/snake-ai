# Snake AI - Project-Based User Data Storage

## ✅ **New Project-Based System**

### **Complete Project Management:**
Snake AI now uses a comprehensive project-based system that organizes all training data, models, and metadata into self-contained project folders.

### **Directory Structure by OS:**

#### **Windows:**
- Projects: `%APPDATA%\SnakeAI\projects\`
- Example: `C:\Users\jross\AppData\Roaming\SnakeAI\projects\`
- Legacy data: `%APPDATA%\SnakeAI\last_model.txt` (still supported)

#### **macOS:**
- Projects: `~/Library/Application Support/SnakeAI/projects/`
- Legacy data: `~/Library/Application Support/SnakeAI/last_model.txt`

#### **Linux:**
- Projects: `~/.config/SnakeAI/projects/`
- Legacy data: `~/.config/SnakeAI/last_model.txt`

#### **Universal Fallback:**
- Projects: `~/.snake_ai/projects/`

## �️ **Project Organization**

### **Each Project Contains:**
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
│   └── (future configurations)
└── project.json              # Project metadata
```

### **Project Metadata (`project.json`):**
- Project name and description
- Creation and modification dates
- Training history and statistics
- Best scores and rewards
- Model architecture information
- Training configuration settings

## 🔧 **Benefits**

### **✅ Complete Organization:**
- All training data in one place per project
- Easy backup and sharing (export as ZIP)
- No scattered files across different directories
- Automatic data management

### **✅ Cross-Platform:**
- Follows OS conventions for user data
- Windows: AppData/Roaming
- Mac: Library/Application Support  
- Linux: .config directory

### **✅ Persistent Storage:**
- Projects persist across application restarts
- Training history preserved
- Easy to resume interrupted training sessions

### **✅ Privacy & Security:**
- User data stays in user space
- No project contamination
- Easy to find and manage
- Projects can be exported/imported securely

## 🎯 **How It Works**

### **Project Creation Process:**
1. User selects "Create New Project"
2. Enters project name and description
3. System creates organized project folder structure
4. Metadata file (`project.json`) initialized
5. Project ready for training

### **Project Loading Process:**
1. User selects "Load Project"
2. System scans projects directory
3. Shows available projects with metadata
4. User selects project to load
5. Training can continue from last checkpoint

### **Training Session Management:**
1. Training data automatically logged to CSV files
2. Best models saved when performance improves
3. Checkpoints saved periodically
4. Training plots generated and saved
5. Project metadata updated with session history

### **Export/Import Process:**
- **Export**: Project folder compressed to ZIP archive
- **Import**: ZIP archive extracted to projects directory
- All data, models, and metadata preserved

## 🛠️ **Technical Details**

### **Project Management Functions:**
- `ProjectManager` - Manages multiple projects
- `SnakeAIProject` - Individual project operations
- `interactive_project_selection()` - User project selection dialog
- Export/import functionality for project archiving

### **Legacy Compatibility:**
- Still supports old `last_model.txt` for migration
- Can load legacy model files
- Automatic conversion to project format when possible

## 📁 **Migration Guide**

### **From Legacy System:**
1. Old models in loose files are still loadable
2. First training session will create project structure
3. Legacy data automatically organized into project format
4. Old files can be safely removed after migration
- `get_last_model_info()` - Debug info about user data

### **File Structure:**
```
%APPDATA%\SnakeAI\
├── last_model.txt          # Absolute path to last loaded model
└── [future: preferences.json]  # Room for future settings
```

### **Security:**
- Only stores file paths, not model data
- Uses standard OS user data locations
- No sensitive information stored

## 🔄 **Migration**

### **From Old System:**
- Old `models/last_model.txt` files are ignored
- New system creates fresh user data
- No data loss - just need to load preferred model once

### **Portable Usage:**
- Each user gets their own preferences
- Project can be shared without user data
- Clean separation of concerns

This implementation follows best practices for desktop applications and provides a much cleaner user experience! 🚀
