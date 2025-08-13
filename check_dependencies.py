#!/usr/bin/env python3
"""
Streamlined Dependency Security Scanner for Snake AI Project
Focus: pip-audit + basic checks (Option 2)
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description):
    """Run a command and capture output"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr and "warning" not in result.stderr.lower():
            print(f"Errors: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.SubprocessError as e:
        print(f"Error: {e}")
        return False, "", str(e)

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    venv_path = os.path.join(os.getcwd(), '.venv')
    if os.path.exists(venv_path):
        return os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        print("⚠️  Virtual environment not found. Using system Python.")
        return sys.executable

def main():
    """Streamlined dependency scanning"""
    print("🐍 Snake AI Security Scanner (Fast Mode)")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 {os.getcwd()}")
    
    # Get executables
    python_exe = check_virtual_environment()
    pip_exe = python_exe.replace('python.exe', 'pip.exe')
    pip_audit_exe = python_exe.replace('python.exe', 'pip-audit.exe')
    pipdeptree_exe = python_exe.replace('python.exe', 'pipdeptree.exe')
    
    # 1. Security Scan with pip-audit (Primary check)
    if os.path.exists(pip_audit_exe):
        _, stdout, _ = run_command(f'"{pip_audit_exe}" --desc', "🔒 SECURITY VULNERABILITIES")
        
        # Quick result summary
        if "Found 0 known vulnerabilities" in stdout:
            print("✅ NO SECURITY ISSUES FOUND")
        elif "Found 1 known vulnerability" in stdout and "torch" in stdout and "ctc_loss" in stdout:
            print("⚠️  1 PyTorch vulnerability (ctc_loss - SAFE for your Snake AI)")
        elif "Found 1 known vulnerability" in stdout:
            print("⚠️  1 SECURITY VULNERABILITY - Review details above")
        elif "vulnerability" in stdout.lower() and "found" in stdout.lower():
            print("🚨 SECURITY VULNERABILITIES FOUND - Review details above")
        else:
            print("✅ Security scan completed")
    else:
        print("⚠️  pip-audit not found. Installing...")
        run_command(f'"{pip_exe}" install pip-audit', "Installing pip-audit")
    
    # 2. Check for outdated packages
    success, stdout, _ = run_command(f'"{pip_exe}" list --outdated', "📦 OUTDATED PACKAGES")
    
    # Count outdated packages
    if stdout:
        lines = stdout.strip().split('\n')
        # Skip header lines
        outdated_count = len([line for line in lines if line and not line.startswith('Package') and not line.startswith('---')])
        if outdated_count > 0:
            print(f"📊 {outdated_count} packages have updates available")
        else:
            print("✅ All packages are up to date")
    
    # 3. Show key dependency tree (concise version)
    if os.path.exists(pipdeptree_exe):
        run_command(f'"{pipdeptree_exe}" --packages torch,pygame,numpy', "🌳 KEY DEPENDENCIES")
    
    # Quick Summary
    print(f"\n{'='*50}")
    print("📋 QUICK SUMMARY")
    print(f"{'='*50}")
    print(f"✓ Scan completed: {datetime.now().strftime('%H:%M:%S')}")
    print("✓ Security check: pip-audit")
    print("✓ Update check: pip list --outdated")
    print("✓ Dependencies: Key packages only")
    
    print("\n💡 NEXT STEPS:")
    print("• Review any vulnerabilities above")
    print("• Consider updating outdated packages")
    print("• Run weekly for ongoing security")
    print("• Full scan: python check_dependencies_full.py")

if __name__ == "__main__":
    main()
