#!/usr/bin/env python3
"""
Interactive script to run documentation cleanup.
Provides a user-friendly interface for executing the cleanup.
"""

import os
from pathlib import Path
import platform
import subprocess
import sys


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def print_header():
    """Print the script header."""
    print("=" * 70)
    print("  Neural DSL - Documentation Cleanup")
    print("=" * 70)
    print()


def print_info():
    """Print information about what will be cleaned."""
    print("This cleanup will remove 67+ redundant documentation files:")
    print()
    print("  • Root level *SUMMARY.md files (14)")
    print("  • .github/ summary files (2)")
    print("  • docs/ redundant files (7)")
    print("  • examples/ files (2)")
    print("  • neural/ module quick-starts (33)")
    print("  • scripts/ files (1)")
    print("  • tests/ files (6)")
    print("  • website/ files (2)")
    print()
    print("PRESERVED FILES:")
    print("  ✓ docs/quickstart.md (core documentation)")
    print("  ✓ docs/quick_reference.md (core documentation)")
    print("  ✓ docs/archive/* (historical documentation)")
    print()


def get_script_path():
    """Determine which cleanup script to use based on the platform."""
    system = platform.system()
    
    if system == "Windows":
        # Prefer .bat for Windows Command Prompt compatibility
        if Path("delete_quick_and_summary_docs.bat").exists():
            return "delete_quick_and_summary_docs.bat"
        elif Path("delete_quick_and_summary_docs.ps1").exists():
            return "delete_quick_and_summary_docs.ps1"
    else:
        # Unix/Linux/macOS
        if Path("delete_quick_and_summary_docs.sh").exists():
            return "delete_quick_and_summary_docs.sh"
    
    return None


def run_cleanup(script_path):
    """Execute the cleanup script."""
    system = platform.system()
    
    try:
        if system == "Windows":
            if script_path.endswith(".ps1"):
                # PowerShell script
                subprocess.run(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path],
                    check=True
                )
            else:
                # Batch script
                subprocess.run([script_path], shell=True, check=True)
        else:
            # Unix/Linux/macOS - make executable and run
            Path(script_path).chmod(0o755)
            subprocess.run([f"./{script_path}"], shell=True, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running cleanup script: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def confirm_action():
    """Ask user to confirm the cleanup action."""
    print("⚠️  WARNING: This will permanently delete 67+ files.")
    print()
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def show_next_steps():
    """Show next steps after cleanup."""
    print()
    print("=" * 70)
    print("  Next Steps")
    print("=" * 70)
    print()
    print("1. Verify the cleanup:")
    print("   git status")
    print()
    print("2. Stage the deletions:")
    print("   git add -A")
    print()
    print("3. Commit the changes:")
    print('   git commit -m "docs: remove 67 redundant QUICK*.md and *SUMMARY.md files"')
    print()
    print("4. (Optional) Remove cleanup scripts:")
    if platform.system() == "Windows":
        print("   del delete_quick_and_summary_docs.*")
        print("   del cleanup_quick_and_summary_docs.py")
        print("   del run_documentation_cleanup.py")
        print("   del QUICK_SUMMARY_CLEANUP_README.md")
        print("   del FILES_TO_DELETE_MANIFEST.txt")
        print("   del DOCUMENTATION_CLEANUP_IMPLEMENTATION.md")
    else:
        print("   rm delete_quick_and_summary_docs.{bat,sh,ps1}")
        print("   rm cleanup_quick_and_summary_docs.py")
        print("   rm run_documentation_cleanup.py")
        print("   rm QUICK_SUMMARY_CLEANUP_README.md")
        print("   rm FILES_TO_DELETE_MANIFEST.txt")
        print("   rm DOCUMENTATION_CLEANUP_IMPLEMENTATION.md")
    print()


def main():
    """Main function."""
    clear_screen()
    print_header()
    print_info()
    
    # Find the appropriate script
    script_path = get_script_path()
    
    if not script_path:
        print("❌ Error: Cleanup script not found!")
        print()
        print("Expected files:")
        print("  - delete_quick_and_summary_docs.bat (Windows)")
        print("  - delete_quick_and_summary_docs.ps1 (Windows)")
        print("  - delete_quick_and_summary_docs.sh (Unix/Linux/macOS)")
        sys.exit(1)
    
    print(f"Using cleanup script: {script_path}")
    print()
    
    # Confirm action
    if not confirm_action():
        print("\n✋ Cleanup cancelled.")
        sys.exit(0)
    
    print()
    print("🚀 Starting cleanup...")
    print()
    
    # Run cleanup
    success = run_cleanup(script_path)
    
    if success:
        print()
        print("✅ Cleanup completed successfully!")
        show_next_steps()
    else:
        print()
        print("❌ Cleanup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
