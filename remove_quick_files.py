#!/usr/bin/env python3
"""Remove all QUICK*.md files after consolidation."""

import os


# List of all QUICK*.md files to remove (from glob output)
QUICK_FILES = [
    "TEST_RESULTS_QUICK_REFERENCE.md",
    "TESTING_QUICK_REFERENCE.md",
    "QUICK_FIXES.md",
    "docs/archive/DISTRIBUTION_QUICK_REF.md",
    "docs/aquarium/QUICK_REFERENCE.md",
    "docs/RELEASE_QUICK_START.md",
    "docs/MARKETING_AUTOMATION_QUICK_REF.md",
    "docs/DEPLOYMENT_QUICK_START.md",
    "docs/mlops/QUICK_REFERENCE.md",
    "docs/quickstart.md",
    "examples/EXAMPLES_QUICK_REF.md",
    "examples/attention_examples/QUICKSTART.md",
    "neural/ai/QUICK_START.md",
    "neural/ai/QUICKSTART.md",
    "neural/api/QUICK_START.md",
    "neural/aquarium/DEBUGGER_QUICKSTART.md",
    "neural/aquarium/EXPORT_QUICK_START.md",
    "neural/aquarium/HPO_QUICKSTART.md",
    "neural/aquarium/PACKAGING_QUICKSTART.md",
    "neural/aquarium/QUICKSTART.md",
    "neural/aquarium/QUICK_REFERENCE.md",
    "neural/aquarium/QUICK_START.md",
    "neural/aquarium/src/components/editor/QUICKSTART.md",
    "neural/aquarium/src/components/terminal/QUICKSTART.md",
    "neural/automl/QUICK_START.md",
    "neural/config/QUICKSTART.md",
    "neural/cost/QUICK_REFERENCE.md",
    "neural/dashboard/QUICKSTART.md",
    "neural/data/QUICKSTART.md",
    "neural/education/QUICK_START.md",
    "neural/federated/QUICKSTART.md",
    "neural/integrations/QUICK_REFERENCE.md",
    "neural/marketplace/QUICK_START.md",
    "neural/monitoring/QUICKSTART.md",
    "neural/no_code/QUICKSTART.md",
    "neural/teams/QUICK_START.md",
    "neural/tracking/QUICK_REFERENCE.md",
    "neural/visualization/QUICKSTART_GALLERY.md",
    "tests/aquarium_e2e/QUICKSTART.md",
    "tests/aquarium_ide/QUICK_START.md",
    "tests/benchmarks/QUICK_REFERENCE.md",
    "tests/integration_tests/QUICK_START.md",
    "tests/performance/QUICK_START.md",
    "website/QUICKSTART.md",
    "website/docs/getting-started/quick-start.md",
]

def main():
    removed_count = 0
    not_found_count = 0
    error_count = 0
    
    print("Removing QUICK*.md files...")
    print("=" * 70)
    
    for filepath in QUICK_FILES:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"✓ Removed: {filepath}")
                removed_count += 1
            else:
                print(f"⚠ Not found: {filepath}")
                not_found_count += 1
        except Exception as e:
            print(f"✗ Error removing {filepath}: {e}")
            error_count += 1
    
    print("=" * 70)
    print("\nSummary:")
    print(f"  Removed: {removed_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files processed: {len(QUICK_FILES)}")

if __name__ == "__main__":
    main()
