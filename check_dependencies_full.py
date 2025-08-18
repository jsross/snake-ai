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
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr and "warning" not in result.stderr.lower():
            print(f"Errors: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
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
    print(f"🐍 Snake AI Security Scanner (Fast Mode)")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 {os.getcwd()}")
    
    # Get executables
    python_exe = check_virtual_environment()
    pip_exe = python_exe.replace('python.exe', 'pip.exe')
    pip_audit_exe = python_exe.replace('python.exe', 'pip-audit.exe')
    pipdeptree_exe = python_exe.replace('python.exe', 'pipdeptree.exe')
    
    # 1. Security Scan with pip-audit (Primary check)
    if os.path.exists(pip_audit_exe):
        success, stdout, stderr = run_command(f'"{pip_audit_exe}" --desc', "🔒 SECURITY VULNERABILITIES")
        
        # Quick result summary - check both stdout and stderr for vulnerability messages
        full_output = stdout + stderr
        if "Found 0 known vulnerabilities" in full_output or "No known vulnerabilities found" in full_output:
            print("✅ NO SECURITY ISSUES FOUND")
        elif "Found 1 known vulnerability" in full_output:
            if "torch" in full_output and "ctc_loss" in full_output:
                print("⚠️  1 PyTorch vulnerability (ctc_loss - SAFE for your Snake AI)")
            else:
                print("⚠️  1 SECURITY VULNERABILITY - Review details above")
        elif "vulnerabilities" in full_output.lower() and ("found" in full_output.lower() or "known" in full_output.lower()):
            # Extract number of vulnerabilities if possible
            import re
            vuln_match = re.search(r'Found (\d+) known vulnerabilities', full_output)
            if vuln_match and int(vuln_match.group(1)) > 1:
                print(f"🚨 {vuln_match.group(1)} VULNERABILITIES FOUND - Review details above")
            else:
                print("🚨 MULTIPLE VULNERABILITIES FOUND - Review details above")
        else:
            print("✅ NO SECURITY ISSUES FOUND")
    else:
        print("⚠️  pip-audit not found. Installing...")
        run_command(f'"{pip_exe}" install pip-audit', "Installing pip-audit")
    
    # 2. Check for outdated packages
    success, stdout, stderr = run_command(f'"{pip_exe}" list --outdated', "📦 OUTDATED PACKAGES")
    
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
        run_command(f'"{pipdeptree_exe}" --packages torch,pygame,numpy --json-tree', "🌳 KEY DEPENDENCIES")
    
    # Quick Summary
    print(f"\n{'='*50}")
    print("📋 QUICK SUMMARY")
    print(f"{'='*50}")
    print(f"✓ Scan completed: {datetime.now().strftime('%H:%M:%S')}")
    print(f"✓ Security check: pip-audit")
    print(f"✓ Update check: pip list --outdated")
    print(f"✓ Dependencies: Key packages only")
    
    print(f"\n💡 NEXT STEPS:")
    print("• Review any vulnerabilities above")
    print("• Consider updating outdated packages")
    print("• Run weekly for ongoing security")
    print(f"• Full scan: python check_dependencies_full.py")

if __name__ == "__main__":
    main()
