# Snake AI Dependency Security Scanning

This directory contains automated tools for regularly checking the security and status of Python dependencies in your Snake AI project.

## 🔧 Setup

### Prerequisites
The essential security scanning tools are already installed in your virtual environment:
- `pip-audit` - Primary security vulnerability scanner
- `pipdeptree` - Dependency tree visualization (optional)

### Installation
If you need to install the tools on a new setup:
```bash
pip install -r requirements-security.txt
```

## 🚀 Usage

### Quick Scan (Recommended)

#### Option 1: Python Script - **FAST MODE**
```bash
python check_dependencies.py
```
**What it does:**
- 🔒 Security vulnerabilities (pip-audit)
- 📦 Outdated packages
- 🌳 Key dependency tree
- ⚡ **Quick results in ~10 seconds**

#### Option 2: Batch File (Windows)
```bash
scan_dependencies.bat
```

#### Option 3: PowerShell Script
```powershell
.\scan_dependencies.ps1
```

### Comprehensive Scan (Full Details)
```bash
python check_dependencies_full.py
```
**What it includes:**
- All quick scan features
- Complete package listing
- Safety database check
- Detailed reporting
- ⏱️ Takes ~30-60 seconds

### Automated Scanning

#### Set up Weekly Scheduled Scan
Run as Administrator:
```bash
setup_scheduled_scan.bat
```

This creates a Windows Task Scheduler job that runs every Monday at 9:00 AM.

## 📊 Current Status Summary

### ✅ **Security Status**
Based on the latest scan:

- **pip-audit**: 1 PyTorch vulnerability (ctc_loss function)
- **Impact**: **NONE** - Your Snake AI project doesn't use the vulnerable function
- **Action**: No immediate action required

### � **Package Status**
- **Total packages**: 95 installed
- **Outdated packages**: 7 packages have updates available
- **Key packages**: torch, pygame, numpy, matplotlib all working

## 🎯 Quick Decision Guide

| Scan Type | When to Use | Time | Best For |
|-----------|-------------|------|----------|
| **Quick Scan** | Weekly, before coding | ~10s | Regular monitoring |
| **Full Scan** | Monthly, before releases | ~60s | Comprehensive review |
| **Scheduled** | Set and forget | Auto | Ongoing protection |

## 🛡️ Security Findings

### PyTorch Vulnerability (GHSA-887c-mr87-cxwp)
- **Package**: torch 2.7.1  
- **Function**: torch.nn.functional.ctc_loss
- **Risk Level**: **LOW** for your project
- **Reason**: Your Snake AI uses Deep Q-Networks, not CTC loss functions
- **Action**: Monitor for updates, but not urgent

## 🔧 Configuration

Edit `dependency_scan_config.ini` to customize scanning behavior.

## 📅 Scheduled Scanning

### View Scheduled Task
```cmd
schtasks /query /tn "SnakeAI_DependencyScan"
```

### Run Scan Now
```cmd
schtasks /run /tn "SnakeAI_DependencyScan"
```

### Remove Scheduled Task
```cmd
schtasks /delete /tn "SnakeAI_DependencyScan" /f
```

## 📂 Files

### Primary Files
- `check_dependencies.py` - **Fast scan** (pip-audit + basics)
- `scan_dependencies.bat` - Windows batch wrapper
- `scan_dependencies.ps1` - PowerShell wrapper

### Additional Files
- `check_dependencies_full.py` - Comprehensive scan
- `setup_scheduled_scan.bat` - Create scheduled task
- `dependency_scan_config.ini` - Configuration file
- `requirements-security.txt` - Security tools requirements

## 🎯 Best Practices

1. **Weekly Quick Scans**: Use `python check_dependencies.py`
2. **Monthly Full Scans**: Use `python check_dependencies_full.py`
3. **Before Releases**: Always scan before deploying
4. **Update Carefully**: Test thoroughly after updating packages
5. **Monitor Critical Packages**: Pay special attention to torch, numpy, pygame

## 🆘 Troubleshooting

### Virtual Environment Issues
Make sure you're in the project directory and the virtual environment is activated:
```bash
cd c:\Users\jross\Source\ai-snake
.venv\Scripts\activate
```

### Permission Issues
Some operations may require Administrator privileges:
- Right-click Command Prompt → "Run as Administrator"
- Then run the setup scripts

### Missing Tools
If scanning tools are missing:
```bash
pip install pip-audit pipdeptree
```

## 📈 Interpreting Results

### 🟢 Green (Safe)
```
✅ NO SECURITY ISSUES FOUND
✅ All packages are up to date
```

### 🟡 Yellow (Attention)
```
⚠️ 1 PyTorch vulnerability (ctc_loss - SAFE for your Snake AI)
📊 7 packages have updates available
```

### 🔴 Red (Action Required)
```
🚨 MULTIPLE VULNERABILITIES FOUND
⚠️ 1 SECURITY VULNERABILITY - Review details above
```

## ⚡ Performance

| Scan Type | Duration | Packages Checked | Detail Level |
|-----------|----------|------------------|--------------|
| Quick     | ~10 sec  | Security + key deps | Essential |
| Full      | ~60 sec  | All packages | Comprehensive |

---

**Last Updated**: August 12, 2025  
**Recommended**: Use quick scan weekly, full scan monthly  
**Next Scan**: August 19, 2025
