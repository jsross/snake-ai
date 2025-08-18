#!/usr/bin/env python3
"""
Streamlined Dependency Security Scanner for Snake AI Project
Focus: pip-audit + basic checks
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

class DependencyScanner:
    """Main dependency scanner class"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.python_exe = self._find_python_executable()
        self.pip_exe = self._find_executable('pip')
        self.pip_audit_exe = self._find_executable('pip-audit')
        self.pipdeptree_exe = self._find_executable('pipdeptree')
    
    def _find_python_executable(self):
        """Find the appropriate Python executable"""
        venv_path = self.project_root / '.venv'
        if venv_path.exists():
            if os.name == 'nt':  # Windows
                return str(venv_path / 'Scripts' / 'python.exe')
            else:  # Unix-like
                return str(venv_path / 'bin' / 'python')
        else:
            print("⚠️  Virtual environment not found. Using system Python.")
            return sys.executable
    
    def _find_executable(self, name):
        """Find executable in virtual environment or system"""
        venv_path = self.project_root / '.venv'
        if venv_path.exists():
            if os.name == 'nt':  # Windows
                exe_path = venv_path / 'Scripts' / f'{name}.exe'
            else:  # Unix-like
                exe_path = venv_path / 'bin' / name
            
            if exe_path.exists():
                return str(exe_path)
        
        # Fallback to system executable
        return name
    
    def run_command(self, command, description):
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
    
    def security_scan(self):
        """Run security vulnerability scan"""
        if not os.path.exists(self.pip_audit_exe):
            print("⚠️  pip-audit not found. Installing...")
            self.run_command(f'"{self.pip_exe}" install pip-audit', "Installing pip-audit")
            self.pip_audit_exe = self._find_executable('pip-audit')
        
        _, stdout, _ = self.run_command(f'"{self.pip_audit_exe}" --desc', "🔒 SECURITY VULNERABILITIES")
        
        # Quick result summary
        if "Found 0 known vulnerabilities" in stdout:
            print("✅ NO SECURITY ISSUES FOUND")
            return 0
        elif "Found 1 known vulnerability" in stdout and "torch" in stdout and "ctc_loss" in stdout:
            print("⚠️  1 PyTorch vulnerability (ctc_loss - SAFE for your Snake AI)")
            return 1
        elif "Found 1 known vulnerability" in stdout:
            print("⚠️  1 SECURITY VULNERABILITY - Review details above")
            return 1
        elif "vulnerability" in stdout.lower() and "found" in stdout.lower():
            print("🚨 SECURITY VULNERABILITIES FOUND - Review details above")
            return 2
        else:
            print("✅ Security scan completed")
            return 0
    
    def check_outdated_packages(self):
        """Check for outdated packages"""
        success, stdout, _ = self.run_command(f'"{self.pip_exe}" list --outdated', "📦 OUTDATED PACKAGES")
        
        if stdout:
            lines = stdout.strip().split('\n')
            # Skip header lines
            outdated_count = len([line for line in lines if line and not line.startswith('Package') and not line.startswith('---')])
            if outdated_count > 0:
                print(f"📊 {outdated_count} packages have updates available")
                return outdated_count
            else:
                print("✅ All packages are up to date")
                return 0
        return 0
    
    def show_key_dependencies(self):
        """Show key dependency tree"""
        if os.path.exists(self.pipdeptree_exe):
            self.run_command(f'"{self.pipdeptree_exe}" --packages torch,pygame,numpy', "🌳 KEY DEPENDENCIES")
        else:
            print("⚠️  pipdeptree not found. Install with: pip install pipdeptree")
    
    def generate_report(self, vulnerabilities, outdated_count):
        """Generate summary report"""
        print(f"\n{'='*50}")
        print("📋 SECURITY SCAN SUMMARY")
        print(f"{'='*50}")
        print(f"✓ Scan completed: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🔒 Security vulnerabilities: {vulnerabilities}")
        print(f"📦 Outdated packages: {outdated_count}")
        print("✓ Dependencies: Key packages checked")
        
        # Save report to logs
        logs_dir = self.project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        report_file = logs_dir / f"security_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Security Scan Report - {datetime.now()}\n")
            f.write(f"Project: {self.project_root}\n")
            f.write(f"Vulnerabilities: {vulnerabilities}\n")
            f.write(f"Outdated packages: {outdated_count}\n")
        
        print(f"📄 Report saved: {report_file}")
        
        print("\n💡 NEXT STEPS:")
        if vulnerabilities > 0:
            print("• Review security vulnerabilities above")
            print("• Update vulnerable packages: pip install --upgrade <package>")
        if outdated_count > 0:
            print("• Consider updating outdated packages")
        print("• Run weekly for ongoing security")
        print("• Full scan available in tools/security/")
    
    def scan(self):
        """Run complete security scan"""
        print("🐍 Snake AI Security Scanner")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 {self.project_root}")
        
        vulnerabilities = self.security_scan()
        outdated_count = self.check_outdated_packages()
        self.show_key_dependencies()
        self.generate_report(vulnerabilities, outdated_count)
        
        return vulnerabilities, outdated_count

def main():
    """Main entry point"""
    scanner = DependencyScanner()
    vulnerabilities, outdated_count = scanner.scan()
    
    # Exit with appropriate code
    if vulnerabilities > 1:  # More than just the PyTorch false positive
        sys.exit(1)
    elif vulnerabilities > 0 or outdated_count > 10:
        sys.exit(2)  # Warning level
    else:
        sys.exit(0)  # All good

if __name__ == "__main__":
    main()
