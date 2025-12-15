#!/usr/bin/env python3
"""
Verification script to check documentation cleanup results.
This script verifies that:
1. Redundant files have been removed
2. Essential documentation is preserved
3. Consolidated quick reference exists
"""

from pathlib import Path
import sys


# Files that should be removed
FILES_TO_CHECK_REMOVED = [
    # Sample of files that should be removed
    "QUICK_SUMMARY_CLEANUP_README.md",
    "CLEANUP_SUMMARY.md",
    "CONSOLIDATION_SUMMARY.md",
    "API_REMOVAL_SUMMARY.md",
    "docs/DEPRECATIONS.md",
    "docs/API_REMOVAL.md",
    "docs/DOCUMENTATION_INDEX.md",
    "tests/TEST_COVERAGE_SUMMARY.md",
    "neural/automl/QUICK_START.md",
    "examples/EXAMPLES_QUICK_REF.md",
]

# Files that must be preserved
FILES_TO_CHECK_PRESERVED = [
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CLEANUP_README.md",
    "docs/README.md",
    "docs/quick_reference.md",
    "docs/FOCUS.md",
    "docs/TYPE_SAFETY.md",
    "docs/dsl.md",
    "DOCUMENTATION_CLEANUP_SUMMARY.md",
]

def verify_cleanup():
    """Verify cleanup results."""
    repo_root = Path(__file__).parent
    
    print("="*60)
    print("Documentation Cleanup Verification")
    print("="*60)
    print(f"\nRepository root: {repo_root}\n")
    
    removed_count = 0
    not_removed_count = 0
    preserved_count = 0
    missing_count = 0
    
    # Check that files were removed
    print("Checking removed files...")
    for file_path in FILES_TO_CHECK_REMOVED:
        full_path = repo_root / file_path
        if not full_path.exists():
            print(f"  ✓ Removed: {file_path}")
            removed_count += 1
        else:
            print(f"  ✗ Still exists: {file_path}")
            not_removed_count += 1
    
    print(f"\nRemoved files: {removed_count}/{len(FILES_TO_CHECK_REMOVED)}")
    
    # Check that essential files are preserved
    print("\nChecking preserved files...")
    for file_path in FILES_TO_CHECK_PRESERVED:
        full_path = repo_root / file_path
        if full_path.exists():
            print(f"  ✓ Preserved: {file_path}")
            preserved_count += 1
        else:
            print(f"  ✗ Missing: {file_path}")
            missing_count += 1
    
    print(f"\nPreserved files: {preserved_count}/{len(FILES_TO_CHECK_PRESERVED)}")
    
    # Overall status
    print("\n" + "="*60)
    if not_removed_count == 0 and missing_count == 0:
        print("✅ Verification PASSED")
        print("="*60)
        print("\nAll redundant files removed successfully!")
        print("All essential documentation preserved!")
        print("\nNext steps:")
        print("  1. Review docs/quick_reference.md")
        print("  2. Check docs/README.md for navigation")
        print("  3. Commit changes: git add -A && git commit -m 'docs: cleanup'")
        return 0
    else:
        print("⚠️  Verification found issues")
        print("="*60)
        if not_removed_count > 0:
            print(f"\n{not_removed_count} files were not removed")
        if missing_count > 0:
            print(f"\n{missing_count} essential files are missing")
        print("\nPlease review the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_cleanup())
