#!/usr/bin/env python3
"""
Master script to execute documentation cleanup.
This script removes redundant documentation files and provides comprehensive reporting.
"""

from pathlib import Path
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: Command not found. Make sure Python is in your PATH.")
        return False

def main():
    """Execute documentation cleanup."""
    repo_root = Path(__file__).parent
    
    print("="*60)
    print("Neural DSL - Documentation Cleanup")
    print("="*60)
    print(f"\nRepository root: {repo_root}")
    print("\nThis script will:")
    print("  1. Remove 60+ redundant documentation files")
    print("  2. Preserve all essential documentation")
    print("  3. Generate cleanup report")
    print("\nEssential docs preserved:")
    print("  - README.md, AGENTS.md, CONTRIBUTING.md, CHANGELOG.md")
    print("  - docs/ (all core documentation)")
    print("  - docs/quick_reference.md (consolidated quick reference)")
    
    # Confirm execution
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nCleanup cancelled.")
        return 0
    
    # Run cleanup
    success = run_command(
        [sys.executable, "cleanup_redundant_docs.py"],
        "Removing redundant documentation files"
    )
    
    if not success:
        print("\n❌ Cleanup encountered errors.")
        return 1
    
    print("\n" + "="*60)
    print("✅ Documentation cleanup completed successfully!")
    print("="*60)
    
    print("\nNext steps:")
    print("  1. Review DOCUMENTATION_CLEANUP_SUMMARY.md for details")
    print("  2. Check docs/quick_reference.md for consolidated information")
    print("  3. Stage changes: git add -A")
    print("  4. Commit: git commit -m 'docs: remove redundant documentation'")
    
    print("\nDocumentation structure:")
    print("  - README.md - Main project documentation")
    print("  - docs/quick_reference.md - Consolidated quick reference ⭐")
    print("  - docs/README.md - Documentation navigation")
    print("  - DOCUMENTATION_CLEANUP_SUMMARY.md - Cleanup details")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
