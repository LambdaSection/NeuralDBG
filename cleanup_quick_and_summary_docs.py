#!/usr/bin/env python3
"""Script to remove QUICK*.md and *SUMMARY.md files from the repository."""

from pathlib import Path


# List of files to remove
files_to_remove = [
    # Root level QUICK files
    "QUICK_FIXES.md",
    "QUICK_FILES_SCRIPTS_README.md",
    "QUICK_FILES_CONSOLIDATION.md",
    
    # Root level SUMMARY files
    "TEST_SUITE_EXECUTIVE_SUMMARY.md",
    "TEST_ANALYSIS_SUMMARY.md",
    "TEAMS_SIMPLIFICATION_SUMMARY.md",
    "LINTING_FIXES_SUMMARY.md",
    "INTEGRATIONS_SIMPLIFICATION_SUMMARY.md",
    "IMPLEMENTATION_SUMMARY.md",
    "HPO_FIXES_SUMMARY.md",
    "FINAL_FIXES_SUMMARY.md",
    "ERROR_SUGGESTIONS_FIX_SUMMARY.md",
    "DOCKER_CONSOLIDATION_SUMMARY.md",
    "DEPENDENCY_FIX_SUMMARY.md",
    "CONSOLIDATION_SUMMARY.md",
    "CLI_CLEANUP_SUMMARY.md",
    "CLEANUP_SUMMARY.md",
    "CACHE_CLEANUP_SUMMARY.md",
    "BUG_FIXES_SUMMARY.md",
    
    # Website
    "website/QUICKSTART.md",
    "website/docs/getting-started/quick-start.md",
    
    # Tests
    "tests/performance/QUICK_START.md",
    "tests/integration_tests/QUICK_START.md",
    "tests/integration_tests/TEST_SUMMARY.md",
    "tests/benchmarks/QUICK_REFERENCE.md",
    "tests/benchmarks/SUMMARY.md",
    "tests/TEST_COVERAGE_SUMMARY.md",
    
    # Scripts
    "scripts/automation/IMPLEMENTATION_SUMMARY.md",
    
    # Neural modules
    "neural/visualization/QUICKSTART_GALLERY.md",
    "neural/visualization/IMPLEMENTATION_SUMMARY.md",
    "neural/tracking/QUICK_REFERENCE.md",
    "neural/tracking/IMPLEMENTATION_SUMMARY.md",
    "neural/teams/QUICK_START.md",
    "neural/profiling/IMPLEMENTATION_SUMMARY.md",
    "neural/parser/REFACTORING_SUMMARY.md",
    "neural/no_code/QUICKSTART.md",
    "neural/monitoring/QUICKSTART.md",
    "neural/monitoring/IMPLEMENTATION_SUMMARY.md",
    "neural/integrations/QUICK_REFERENCE.md",
    "neural/education/QUICK_START.md",
    "neural/education/IMPLEMENTATION_SUMMARY.md",
    "neural/data/QUICKSTART.md",
    "neural/dashboard/QUICKSTART.md",
    "neural/cost/QUICK_REFERENCE.md",
    "neural/config/QUICKSTART.md",
    "neural/config/SUMMARY.md",
    "neural/automl/QUICK_START.md",
    
    # Aquarium
    "neural/aquarium/src/components/terminal/QUICKSTART.md",
    "neural/aquarium/src/components/terminal/IMPLEMENTATION_SUMMARY.md",
    "neural/aquarium/src/components/editor/QUICKSTART.md",
    "neural/aquarium/src/components/editor/SUMMARY.md",
    "neural/aquarium/src/components/debugger/IMPLEMENTATION_SUMMARY.md",
    "neural/aquarium/QUICK_START.md",
    "neural/aquarium/QUICK_REFERENCE.md",
    "neural/aquarium/QUICKSTART.md",
    "neural/aquarium/PROJECT_SUMMARY.md",
    "neural/aquarium/PLUGIN_IMPLEMENTATION_SUMMARY.md",
    "neural/aquarium/PACKAGING_SUMMARY.md",
    "neural/aquarium/IMPLEMENTATION_SUMMARY.md",
    "neural/aquarium/EXPORT_IMPLEMENTATION_SUMMARY.md",
    
    # Examples
    "examples/attention_examples/QUICKSTART.md",
    "examples/IMPLEMENTATION_SUMMARY.md",
    
    # Docs
    "docs/TRANSFORMER_DOCS_SUMMARY.md",
    "docs/CONSOLIDATION_SUMMARY.md",
    "docs/mlops/QUICK_REFERENCE.md",
    "docs/aquarium/QUICK_REFERENCE.md",
    "docs/aquarium/IMPLEMENTATION_SUMMARY.md",
    "docs/aquarium/DOCUMENTATION_CONSOLIDATION_SUMMARY.md",
]

def main():
    """Remove all QUICK*.md and *SUMMARY.md files."""
    removed_count = 0
    not_found_count = 0
    
    for file_path in files_to_remove:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                full_path.unlink()
                print(f"✓ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Error removing {file_path}: {e}")
        else:
            print(f"- Not found: {file_path}")
            not_found_count += 1
    
    print("\nSummary:")
    print(f"  Removed: {removed_count} files")
    print(f"  Not found: {not_found_count} files")
    print(f"  Total: {len(files_to_remove)} files processed")

if __name__ == "__main__":
    main()
