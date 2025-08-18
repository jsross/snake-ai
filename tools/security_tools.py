#!/usr/bin/env python3
"""
Security scanning utilities for Snake AI project
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_quick_scan():
    """Run a quick security scan"""
    scanner_path = Path(__file__).parent / "security" / "dependency_scanner.py"
    subprocess.run([sys.executable, str(scanner_path)], check=False)

def run_full_scan():
    """Run a comprehensive security scan"""
    print("🔍 Running comprehensive security scan...")
    
    # Run dependency scanner
    run_quick_scan()
    
    # Run bandit for code security issues
    print("\n" + "="*50)
    print("🔒 STATIC CODE ANALYSIS (BANDIT)")
    print("="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "bandit", 
            "-r", "src/", 
            "-f", "txt",
            "--skip", "B101"  # Skip assert_used test
        ], check=False)
    except FileNotFoundError:
        print("⚠️  Bandit not found. Install with: pip install bandit")

def install_security_tools():
    """Install all security scanning tools"""
    print("📦 Installing security tools...")
    requirements_path = Path(__file__).parent.parent / "requirements" / "security.txt"
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "-r", str(requirements_path)
    ], check=False)

def main():
    parser = argparse.ArgumentParser(description="Snake AI Security Scanner")
    parser.add_argument(
        "--quick", "-q", 
        action="store_true", 
        help="Run quick dependency scan only"
    )
    parser.add_argument(
        "--full", "-f", 
        action="store_true", 
        help="Run comprehensive security scan"
    )
    parser.add_argument(
        "--install", "-i", 
        action="store_true", 
        help="Install security tools"
    )
    
    args = parser.parse_args()
    
    if args.install:
        install_security_tools()
    elif args.full:
        run_full_scan()
    elif args.quick:
        run_quick_scan()
    else:
        # Default to quick scan
        run_quick_scan()

if __name__ == "__main__":
    main()
