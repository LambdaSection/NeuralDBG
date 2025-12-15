#!/usr/bin/env python3
"""
Helper script to remove deprecated QUICK*.md files after transition period.

This script should be run after users have had time to transition to the
new consolidated quick reference (docs/quick_reference.md).

Recommended: Wait 3-6 months after deprecation before running this.
"""

import glob
import os


# Files to KEEP (not delete)
KEEP_FILES = {
    "docs/quick_reference.md",  # The new consolidated guide
    "QUICK_FILES_CONSOLIDATION.md",  # Documentation of consolidation
    "docs/archive/QUICK_FILES_CLEANUP_2025.md",  # Archive note
    "CONSOLIDATION_SUMMARY.md",  # Summary document
}

def find_deprecated_quick_files():
    """Find all QUICK*.md files that contain DEPRECATED marker."""
    quick_files = glob.glob("**/*QUICK*.md", recursive=True)
    deprecated = []
    
    for filepath in quick_files:
        # Skip files we want to keep
        normalized_path = filepath.replace("\\", "/")
        if normalized_path in KEEP_FILES:
            continue
            
        # Check if file contains DEPRECATED marker
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'DEPRECATED' in content[:200]:  # Check first 200 chars
                    deprecated.append(filepath)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    
    return sorted(deprecated)

def main():
    print("=" * 70)
    print("Neural DSL - Remove Deprecated QUICK Files")
    print("=" * 70)
    print()
    print("This script will DELETE all deprecated QUICK*.md files.")
    print("Files to keep:", ", ".join(KEEP_FILES))
    print()
    
    # Find deprecated files
    deprecated_files = find_deprecated_quick_files()
    
    if not deprecated_files:
        print("No deprecated QUICK*.md files found.")
        return
    
    print(f"Found {len(deprecated_files)} deprecated files:\n")
    for i, filepath in enumerate(deprecated_files, 1):
        print(f"  {i:2d}. {filepath}")
    
    print("\n" + "=" * 70)
    print("\nWARNING: This will permanently delete these files!")
    print("Make sure you have:")
    print("  1. Waited sufficient time for users to transition (3-6 months)")
    print("  2. Updated any external documentation")
    print("  3. Made a backup or committed current state to git")
    print()
    
    response = input("Do you want to proceed with deletion? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\nCancelled. No files were deleted.")
        return
    
    print("\nDeleting files...")
    print("=" * 70)
    
    deleted_count = 0
    error_count = 0
    
    for filepath in deprecated_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"✓ Deleted: {filepath}")
                deleted_count += 1
            else:
                print(f"⚠ Not found: {filepath}")
        except Exception as e:
            print(f"✗ Error deleting {filepath}: {e}")
            error_count += 1
    
    print("=" * 70)
    print("\nSummary:")
    print(f"  Deleted: {deleted_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total processed: {len(deprecated_files)}")
    
    if deleted_count > 0:
        print("\n" + "=" * 70)
        print("SUCCESS! Deprecated files have been removed.")
        print("\nNext steps:")
        print("  1. Verify the repository still works correctly")
        print("  2. Commit the changes:")
        print("     git add -A")
        print("     git commit -m 'Remove deprecated QUICK reference files'")
        print("  3. Update CONSOLIDATION_SUMMARY.md if needed")
        print("=" * 70)

if __name__ == "__main__":
    main()
