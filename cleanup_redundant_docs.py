#!/usr/bin/env python3
"""
Cleanup script to remove redundant documentation files.
This script removes:
- QUICK_*.md files (except docs/quick_reference.md and docs/quickstart.md)
- *_SUMMARY.md files
- *_IMPLEMENTATION*.md files
- *_COMPLETE.md files
- Obsolete cleanup and consolidation documentation
"""

from pathlib import Path


# Files to delete
FILES_TO_DELETE = [
    # Root level - SUMMARY files
    "API_REMOVAL_SUMMARY.md",
    "BENCHMARKING_IMPLEMENTATION_SUMMARY.md",
    "CACHE_CLEANUP_SUMMARY.md",
    "CLI_CLEANUP_SUMMARY.md",
    "CLEANUP_SUMMARY.md",
    "CONSOLIDATION_SUMMARY.md",
    "DOCKER_CONSOLIDATION_SUMMARY.md",
    "EXAMPLES_IMPLEMENTATION_SUMMARY.md",
    "INTEGRATIONS_SIMPLIFICATION_SUMMARY.md",
    "LOGGING_MIGRATION_SUMMARY.md",
    "TEAMS_SIMPLIFICATION_SUMMARY.md",
    
    # Root level - IMPLEMENTATION/COMPLETE files
    "AQUARIUM_IDE_REMOVAL_IMPLEMENTATION.md",
    "BUG_FIXES_COMPLETE.md",
    "DOCUMENTATION_CLEANUP_COMPLETE.md",
    "DOCUMENTATION_CLEANUP_IMPLEMENTATION.md",
    "DOCUMENTATION_CONSOLIDATION_COMPLETE.md",
    "GITHUB_PAGES_IMPLEMENTATION.md",
    "LOGGING_IMPLEMENTATION.md",
    "TEST_FIXES_IMPLEMENTATION.md",
    "V0.4.0_IMPLEMENTATION_COMPLETE.md",
    "V0.4.0_RELEASE_PREPARATION_COMPLETE.md",
    
    # Root level - QUICK/CLEANUP files
    "QUICK_SUMMARY_CLEANUP_README.md",
    "CLEANUP_NOTES.md",
    "CLEANUP_QUICK_REFERENCE.md",
    "CLEANUP_SCRIPTS_INDEX.md",
    "POST_CLEANUP_TEST_FIXES.md",
    
    # Root level - Status/tracking files
    "V0.4.0_REFACTORING_STATUS.md",
    "TEST_SUITE_RESULTS.md",
    "CONSOLIDATION_CHECKLIST.md",
    "LOGGING_VERIFICATION.md",
    "LOGGING_README.md",
    "INFRASTRUCTURE_CONSOLIDATION.md",
    "NEXT_STEPS.md",
    
    # tests/ directory
    "tests/TEST_COVERAGE_SUMMARY.md",
    "tests/benchmarks/QUICK_REFERENCE.md",
    "tests/integration_tests/QUICK_START.md",
    "tests/integration_tests/INDEX.md",
    "tests/performance/QUICK_START.md",
    
    # neural/ subdirectories - QUICK files
    "neural/automl/QUICK_START.md",
    "neural/config/QUICKSTART.md",
    "neural/dashboard/QUICKSTART.md",
    "neural/data/QUICKSTART.md",
    "neural/education/QUICK_START.md",
    "neural/integrations/QUICK_REFERENCE.md",
    "neural/no_code/QUICKSTART.md",
    "neural/teams/QUICK_START.md",
    "neural/tracking/QUICK_REFERENCE.md",
    "neural/visualization/QUICKSTART_GALLERY.md",
    
    # neural/ subdirectories - IMPLEMENTATION/COMPLETE files
    "neural/benchmarks/IMPLEMENTATION_COMPLETE.md",
    
    # examples/ directory
    "examples/attention_examples/QUICKSTART.md",
    "examples/EXAMPLES_QUICK_REF.md",
    
    # website/ directory
    "website/QUICKSTART.md",
    
    # docs/ directory - QUICK/SUMMARY files (except core quick_reference.md)
    "docs/archive/QUICK_SUMMARY_CLEANUP_2025.md",
    "docs/mlops/QUICK_REFERENCE.md",
    
    # docs/ directory - Marketing/automation files
    "docs/MARKETING_AUTOMATION_DIAGRAM.md",
    "docs/MARKETING_AUTOMATION_GUIDE.md",
    "docs/MARKETING_AUTOMATION_QUICK_REF.md",
    "docs/MARKETING_AUTOMATION_SETUP.md",
    "docs/MARKET_POSITIONING.md",
    "docs/POST_RELEASE_AUTOMATION.md",
    "docs/AUTOMATION_REFERENCE.md",
    
    # docs/ directory - Deployment quick starts
    "docs/DEPLOYMENT_QUICK_START.md",
    "docs/RELEASE_QUICK_START.md",
    
    # docs/ directory - Implementation/setup guides
    "docs/BUILD_DOCS.md",
    "docs/CONFIGURATION_VALIDATION.md",
    "docs/COST_OPTIMIZATION.md",
    "docs/PROFILING_GUIDE.md",
    "docs/SECURITY_SETUP.md",
    "docs/RELEASE_WORKFLOW.md",
    "docs/ERROR_SUGGESTIONS_REFERENCE.md",
    "docs/EXPERIMENT_TRACKING_GUIDE.md",
    "docs/LOGGING_GUIDE.md",
    "docs/DOCSTRING_GUIDE.md",
    "docs/README_CLEANUP.md",
    
    # docs/ directory - Removal/deprecation summaries
    "docs/API_REMOVAL.md",
    "docs/AQUARIUM_IDE_REMOVAL.md",
    "docs/TEAMS_MODULE_REMOVAL.md",
    "docs/DEPRECATIONS.md",
    
    # docs/ directory - Documentation index (redundant with README.md)
    "docs/DOCUMENTATION_INDEX.md",
    
    # scripts/ directory
    "scripts/README_LINTING.md",
    "scripts/CLEANUP_README.md",
]

def delete_files():
    """Delete all redundant documentation files."""
    repo_root = Path(__file__).parent
    deleted_count = 0
    not_found_count = 0
    
    print("Starting documentation cleanup...")
    print(f"Repository root: {repo_root}\n")
    
    for file_path in FILES_TO_DELETE:
        full_path = repo_root / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                print(f"✓ Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {file_path}: {e}")
        else:
            print(f"  Skipped (not found): {file_path}")
            not_found_count += 1
    
    print(f"\n{'='*60}")
    print("Cleanup complete!")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Files not found: {not_found_count}")
    print(f"  Total processed: {len(FILES_TO_DELETE)}")
    print(f"{'='*60}\n")
    
    print("Essential documentation preserved:")
    print("  - README.md (main project documentation)")
    print("  - AGENTS.md (development guide)")
    print("  - CONTRIBUTING.md (contribution guidelines)")
    print("  - CHANGELOG.md (version history)")
    print("  - CLEANUP_README.md (cache cleanup guide)")
    print("  - docs/ (all core documentation)")
    print("  - docs/quick_reference.md (consolidated quick reference)")

if __name__ == "__main__":
    delete_files()
