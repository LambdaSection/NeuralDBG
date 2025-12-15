#!/usr/bin/env python3
"""
List remaining QUICK_*.md and *_SUMMARY.md files after cleanup.
This helps verify that only intended documentation remains.
"""

from pathlib import Path
import re


def find_files(pattern):
    """Find files matching a pattern."""
    repo_root = Path(__file__).parent
    files = []
    
    for path in repo_root.rglob("*.md"):
        # Skip hidden directories and node_modules
        if any(part.startswith('.') for part in path.parts):
            continue
        if 'node_modules' in path.parts:
            continue
        
        filename = path.name
        if re.search(pattern, filename, re.IGNORECASE):
            relative_path = path.relative_to(repo_root)
            files.append(str(relative_path))
    
    return sorted(files)

def main():
    """List remaining documentation files."""
    print("="*60)
    print("Remaining Documentation Files")
    print("="*60)
    
    # Find QUICK_*.md files
    print("\nQUICK_*.md files:")
    quick_files = find_files(r"QUICK.*\.md$")
    if quick_files:
        for f in quick_files:
            print(f"  - {f}")
        print(f"\nTotal: {len(quick_files)} files")
        
        # Check if only expected files remain
        expected_quick = ["docs/quick_reference.md"]
        unexpected = [f for f in quick_files if f.replace("\\", "/") not in expected_quick]
        
        if unexpected:
            print("\n⚠️  Unexpected QUICK_*.md files found:")
            for f in unexpected:
                print(f"     {f}")
        else:
            print("\n✓ Only expected quick reference files remain")
    else:
        print("  None found")
    
    # Find *_SUMMARY.md files
    print("\n" + "-"*60)
    print("\n*_SUMMARY.md files:")
    summary_files = find_files(r".*SUMMARY\.md$")
    if summary_files:
        for f in summary_files:
            print(f"  - {f}")
        print(f"\nTotal: {len(summary_files)} files")
        
        # Check if only expected files remain
        expected_summary = ["DOCUMENTATION_CLEANUP_SUMMARY.md"]
        unexpected = [f for f in summary_files if f.replace("\\", "/") not in expected_summary]
        
        if unexpected:
            print("\n⚠️  Unexpected *_SUMMARY.md files found:")
            for f in unexpected:
                print(f"     {f}")
        else:
            print("\n✓ Only expected summary files remain")
    else:
        print("  None found")
    
    # Find QUICKSTART*.md files
    print("\n" + "-"*60)
    print("\nQUICKSTART*.md files:")
    quickstart_files = find_files(r"QUICKSTART.*\.md$")
    if quickstart_files:
        for f in quickstart_files:
            print(f"  - {f}")
        print(f"\nTotal: {len(quickstart_files)} files")
        
        # All QUICKSTART files should be removed
        if quickstart_files:
            print("\n⚠️  QUICKSTART files should be removed:")
            for f in quickstart_files:
                print(f"     {f}")
    else:
        print("  None found")
        print("\n✓ All QUICKSTART files removed")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    unexpected_count = 0
    if quick_files:
        expected_quick = ["docs\\quick_reference.md", "docs/quick_reference.md"]
        unexpected_count += len([f for f in quick_files if f not in expected_quick])
    
    if summary_files:
        expected_summary = ["DOCUMENTATION_CLEANUP_SUMMARY.md"]
        unexpected_count += len([f for f in summary_files if f not in expected_summary])
    
    unexpected_count += len(quickstart_files) if quickstart_files else 0
    
    if unexpected_count == 0:
        print("\n✅ Documentation cleanup verification PASSED")
        print("\nOnly expected files remain:")
        print("  - docs/quick_reference.md (consolidated quick reference)")
        print("  - DOCUMENTATION_CLEANUP_SUMMARY.md (cleanup summary)")
    else:
        print(f"\n⚠️  Found {unexpected_count} unexpected documentation files")
        print("Review the list above for files that may need removal.")

if __name__ == "__main__":
    main()
