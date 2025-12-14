#!/usr/bin/env python3
"""Script to clean up redundant documentation files."""
import os
import shutil
from pathlib import Path

# Create archive directory if it doesn't exist
archive_dir = Path("docs/archive")
archive_dir.mkdir(parents=True, exist_ok=True)

# List of redundant documentation files to archive
redundant_docs = [
    "AQUARIUM_IMPLEMENTATION_SUMMARY.md",
    "AUTOMATION_GUIDE.md",
    "BENCHMARKS_IMPLEMENTATION_SUMMARY.md",
    "BUG_FIXES.md",
    "CHANGES_SUMMARY.md",
    "CLEANUP_PLAN.md",
    "CLOUD_IMPROVEMENTS_SUMMARY.md",
    "COST_OPTIMIZATION_IMPLEMENTATION.md",
    "DATA_VERSIONING_IMPLEMENTATION.md",
    "DEPENDENCY_CHANGES.md",
    "DEPENDENCY_OPTIMIZATION_SUMMARY.md",
    "DEPLOYMENT_FEATURES.md",
    "DISTRIBUTION_JOURNAL.md",
    "DISTRIBUTION_PLAN.md",
    "DISTRIBUTION_QUICK_REF.md",
    "DOCUMENTATION_SUMMARY.md",
    "EXTRACTED_PROJECTS.md",
    "GITHUB_PUBLISHING_GUIDE.md",
    "GITHUB_RELEASE_v0.3.0.md",
    "IMPLEMENTATION_CHECKLIST.md",
    "IMPLEMENTATION_COMPLETE.md",
    "IMPLEMENTATION_SUMMARY.md",
    "IMPORT_REFACTOR.md",
    "INTEGRATION_IMPLEMENTATION.md",
    "INTEGRATIONS_SUMMARY.md",
    "MARKETPLACE_IMPLEMENTATION.md",
    "MARKETPLACE_SUMMARY.md",
    "MIGRATION_GUIDE_DEPENDENCIES.md",
    "MIGRATION_v0.3.0.md",
    "MLOPS_IMPLEMENTATION.md",
    "MULTIHEADATTENTION_IMPLEMENTATION.md",
    "NEURAL_API_IMPLEMENTATION.md",
    "PERFORMANCE_IMPLEMENTATION.md",
    "POSITIONAL_ENCODING_IMPLEMENTATION.md",
    "POST_RELEASE_AUTOMATION_QUICK_REF.md",
    "POST_RELEASE_IMPLEMENTATION_SUMMARY.md",
    "QUICK_START_AUTOMATION.md",
    "RELEASE_NOTES_v0.3.0.md",
    "RELEASE_VERIFICATION_v0.3.0.md",
    "REPOSITORY_STRUCTURE.md",
    "SETUP_STATUS.md",
    "TEAMS_IMPLEMENTATION.md",
    "TRANSFORMER_DECODER_IMPLEMENTATION.md",
    "TRANSFORMER_ENHANCEMENTS.md",
    "TRANSFORMER_QUICK_REFERENCE.md",
    "V0.3.0_RELEASE_SUMMARY.md",
    "WEBSITE_IMPLEMENTATION_SUMMARY.md",
    "WEBSITE_README.md",
    "ERROR_MESSAGES_GUIDE.md",
]

# Archive redundant files
print("Archiving redundant documentation files...")
for doc_file in redundant_docs:
    src = Path(doc_file)
    if src.exists():
        dst = archive_dir / doc_file
        shutil.move(str(src), str(dst))
        print(f"  Moved: {doc_file} -> docs/archive/")

# List of files to delete completely
files_to_delete = [
    "repro_parser.py",
    "reproduce_issue.py",
    "_install_dev.py",
    "_setup_repo.py",
    "install.bat",
    "install_dev.bat",
    "install_deps.py",
]

print("\nDeleting obsolete files...")
for file_path in files_to_delete:
    path = Path(file_path)
    if path.exists():
        path.unlink()
        print(f"  Deleted: {file_path}")

print("\n✅ Cleanup complete!")
