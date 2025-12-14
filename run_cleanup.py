#!/usr/bin/env python3
"""Master cleanup script to execute repository cleanup."""
import subprocess
import sys

def run_script(script_name):
    """Run a cleanup script and report results."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Execute all cleanup scripts."""
    print("="*60)
    print("NEURAL DSL REPOSITORY CLEANUP")
    print("="*60)
    print("\nThis script will:")
    print("  1. Archive 50+ redundant documentation files")
    print("  2. Remove obsolete development scripts")
    print("  3. Consolidate GitHub Actions workflows (20 -> 4)")
    print("\nFiles will be preserved in docs/archive/")
    
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cleanup cancelled.")
        return
    
    # Run cleanup scripts
    scripts = [
        'cleanup_redundant_files.py',
        'cleanup_workflows.py'
    ]
    
    results = []
    for script in scripts:
        success = run_script(script)
        results.append((script, success))
    
    # Summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    
    for script, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {script}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✅ Repository cleanup completed successfully!")
        print("\nNext steps:")
        print("  1. Review changes with: git status")
        print("  2. Check archived files: ls docs/archive/")
        print("  3. Verify workflows: ls .github/workflows/")
        print("  4. Commit changes: git add -A && git commit -m 'Clean up repository structure'")
    else:
        print("\n⚠️  Some cleanup steps failed. Please review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
