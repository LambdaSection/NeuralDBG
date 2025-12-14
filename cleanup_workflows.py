#!/usr/bin/env python3
"""Script to clean up redundant GitHub Actions workflows."""
import os
from pathlib import Path

# List of workflow files to remove (keeping only essential ones)
workflows_to_remove = [
    ".github/workflows/aquarium-release.yml",
    ".github/workflows/automated_release.yml",
    ".github/workflows/benchmarks.yml",
    ".github/workflows/ci.yml",  # Replaced by essential-ci.yml
    ".github/workflows/close-fixed-issues.yml",
    ".github/workflows/complexity.yml",
    ".github/workflows/metrics.yml",
    ".github/workflows/periodic_tasks.yml",
    ".github/workflows/post_release.yml",
    ".github/workflows/pre-commit.yml",
    ".github/workflows/pylint.yml",  # Covered in essential-ci.yml
    ".github/workflows/pypi.yml",  # Replaced by release.yml
    ".github/workflows/pytest-to-issues.yml",
    ".github/workflows/python-publish.yml",  # Replaced by release.yml
    ".github/workflows/security-audit.yml",  # Consolidated into essential-ci.yml
    ".github/workflows/security.yml",  # Consolidated into essential-ci.yml
    ".github/workflows/snyk-security.yml",
    ".github/workflows/validate_examples.yml",  # Replaced by validate-examples.yml
    ".github/workflows/README_POST_RELEASE.md",  # Obsolete documentation
]

print("Removing redundant GitHub Actions workflows...")
for workflow_file in workflows_to_remove:
    path = Path(workflow_file)
    if path.exists():
        path.unlink()
        print(f"  Removed: {workflow_file}")

print("\n✅ Workflow cleanup complete!")
print("\nKept workflows:")
print("  - essential-ci.yml (lint, test, security)")
print("  - release.yml (PyPI & GitHub releases)")
print("  - codeql.yml (CodeQL security analysis)")
print("  - validate-examples.yml (example validation)")
print("  - README.md (workflow documentation)")
print("  - .env.example (environment template)")
