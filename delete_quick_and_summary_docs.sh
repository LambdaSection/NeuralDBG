#!/bin/bash
# Shell script to delete QUICK*.md and *SUMMARY.md documentation files
# Run this script from the repository root

echo "Deleting QUICK*.md and *SUMMARY.md files..."
echo ""

# Root level files
rm -f "BUG_FIXES_SUMMARY.md" && echo "Deleted: BUG_FIXES_SUMMARY.md"
rm -f "CACHE_CLEANUP_SUMMARY.md" && echo "Deleted: CACHE_CLEANUP_SUMMARY.md"
rm -f "CLEANUP_SUMMARY.md" && echo "Deleted: CLEANUP_SUMMARY.md"
rm -f "CLI_CLEANUP_SUMMARY.md" && echo "Deleted: CLI_CLEANUP_SUMMARY.md"
rm -f "CONSOLIDATION_SUMMARY.md" && echo "Deleted: CONSOLIDATION_SUMMARY.md"
rm -f "DEPENDENCY_FIX_SUMMARY.md" && echo "Deleted: DEPENDENCY_FIX_SUMMARY.md"
rm -f "DOCKER_CONSOLIDATION_SUMMARY.md" && echo "Deleted: DOCKER_CONSOLIDATION_SUMMARY.md"
rm -f "ERROR_SUGGESTIONS_FIX_SUMMARY.md" && echo "Deleted: ERROR_SUGGESTIONS_FIX_SUMMARY.md"
rm -f "FINAL_FIXES_SUMMARY.md" && echo "Deleted: FINAL_FIXES_SUMMARY.md"
rm -f "HPO_FIXES_SUMMARY.md" && echo "Deleted: HPO_FIXES_SUMMARY.md"
rm -f "IMPLEMENTATION_SUMMARY.md" && echo "Deleted: IMPLEMENTATION_SUMMARY.md"
rm -f "INTEGRATIONS_SIMPLIFICATION_SUMMARY.md" && echo "Deleted: INTEGRATIONS_SIMPLIFICATION_SUMMARY.md"
rm -f "LINTING_FIXES_SUMMARY.md" && echo "Deleted: LINTING_FIXES_SUMMARY.md"
rm -f "TEAMS_SIMPLIFICATION_SUMMARY.md" && echo "Deleted: TEAMS_SIMPLIFICATION_SUMMARY.md"

# .github directory
rm -f ".github/MARKETING_AUTOMATION_SUMMARY.md" && echo "Deleted: .github/MARKETING_AUTOMATION_SUMMARY.md"
rm -f ".github/SECURITY_IMPLEMENTATION_SUMMARY.md" && echo "Deleted: .github/SECURITY_IMPLEMENTATION_SUMMARY.md"

# docs directory
rm -f "docs/CONSOLIDATION_SUMMARY.md" && echo "Deleted: docs/CONSOLIDATION_SUMMARY.md"
rm -f "docs/TRANSFORMER_DOCS_SUMMARY.md" && echo "Deleted: docs/TRANSFORMER_DOCS_SUMMARY.md"

# docs/aquarium directory
rm -f "docs/aquarium/DOCUMENTATION_CONSOLIDATION_SUMMARY.md" && echo "Deleted: docs/aquarium/DOCUMENTATION_CONSOLIDATION_SUMMARY.md"
rm -f "docs/aquarium/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: docs/aquarium/IMPLEMENTATION_SUMMARY.md"
rm -f "docs/aquarium/QUICK_REFERENCE.md" && echo "Deleted: docs/aquarium/QUICK_REFERENCE.md"

# docs/mlops directory
rm -f "docs/mlops/QUICK_REFERENCE.md" && echo "Deleted: docs/mlops/QUICK_REFERENCE.md"

# examples directory
rm -f "examples/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: examples/IMPLEMENTATION_SUMMARY.md"
rm -f "examples/attention_examples/QUICKSTART.md" && echo "Deleted: examples/attention_examples/QUICKSTART.md"

# neural/aquarium directory
rm -f "neural/aquarium/EXPORT_IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/aquarium/EXPORT_IMPLEMENTATION_SUMMARY.md"
rm -f "neural/aquarium/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/aquarium/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/aquarium/PACKAGING_SUMMARY.md" && echo "Deleted: neural/aquarium/PACKAGING_SUMMARY.md"
rm -f "neural/aquarium/PLUGIN_IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/aquarium/PLUGIN_IMPLEMENTATION_SUMMARY.md"
rm -f "neural/aquarium/PROJECT_SUMMARY.md" && echo "Deleted: neural/aquarium/PROJECT_SUMMARY.md"
rm -f "neural/aquarium/QUICK_REFERENCE.md" && echo "Deleted: neural/aquarium/QUICK_REFERENCE.md"
rm -f "neural/aquarium/QUICK_START.md" && echo "Deleted: neural/aquarium/QUICK_START.md"
rm -f "neural/aquarium/QUICKSTART.md" && echo "Deleted: neural/aquarium/QUICKSTART.md"

# neural/aquarium/src/components subdirectories
rm -f "neural/aquarium/src/components/debugger/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/aquarium/src/components/debugger/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/aquarium/src/components/editor/QUICKSTART.md" && echo "Deleted: neural/aquarium/src/components/editor/QUICKSTART.md"
rm -f "neural/aquarium/src/components/editor/SUMMARY.md" && echo "Deleted: neural/aquarium/src/components/editor/SUMMARY.md"
rm -f "neural/aquarium/src/components/terminal/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/aquarium/src/components/terminal/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/aquarium/src/components/terminal/QUICKSTART.md" && echo "Deleted: neural/aquarium/src/components/terminal/QUICKSTART.md"

# neural subdirectories
rm -f "neural/automl/QUICK_START.md" && echo "Deleted: neural/automl/QUICK_START.md"
rm -f "neural/config/QUICKSTART.md" && echo "Deleted: neural/config/QUICKSTART.md"
rm -f "neural/config/SUMMARY.md" && echo "Deleted: neural/config/SUMMARY.md"
rm -f "neural/cost/QUICK_REFERENCE.md" && echo "Deleted: neural/cost/QUICK_REFERENCE.md"
rm -f "neural/dashboard/QUICKSTART.md" && echo "Deleted: neural/dashboard/QUICKSTART.md"
rm -f "neural/data/QUICKSTART.md" && echo "Deleted: neural/data/QUICKSTART.md"
rm -f "neural/education/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/education/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/education/QUICK_START.md" && echo "Deleted: neural/education/QUICK_START.md"
rm -f "neural/integrations/QUICK_REFERENCE.md" && echo "Deleted: neural/integrations/QUICK_REFERENCE.md"
rm -f "neural/monitoring/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/monitoring/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/monitoring/QUICKSTART.md" && echo "Deleted: neural/monitoring/QUICKSTART.md"
rm -f "neural/no_code/QUICKSTART.md" && echo "Deleted: neural/no_code/QUICKSTART.md"
rm -f "neural/parser/REFACTORING_SUMMARY.md" && echo "Deleted: neural/parser/REFACTORING_SUMMARY.md"
rm -f "neural/profiling/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/profiling/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/teams/QUICK_START.md" && echo "Deleted: neural/teams/QUICK_START.md"
rm -f "neural/tracking/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/tracking/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/tracking/QUICK_REFERENCE.md" && echo "Deleted: neural/tracking/QUICK_REFERENCE.md"
rm -f "neural/visualization/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: neural/visualization/IMPLEMENTATION_SUMMARY.md"
rm -f "neural/visualization/QUICKSTART_GALLERY.md" && echo "Deleted: neural/visualization/QUICKSTART_GALLERY.md"

# scripts directory
rm -f "scripts/automation/IMPLEMENTATION_SUMMARY.md" && echo "Deleted: scripts/automation/IMPLEMENTATION_SUMMARY.md"

# tests directory
rm -f "tests/TEST_COVERAGE_SUMMARY.md" && echo "Deleted: tests/TEST_COVERAGE_SUMMARY.md"
rm -f "tests/benchmarks/QUICK_REFERENCE.md" && echo "Deleted: tests/benchmarks/QUICK_REFERENCE.md"
rm -f "tests/benchmarks/SUMMARY.md" && echo "Deleted: tests/benchmarks/SUMMARY.md"
rm -f "tests/integration_tests/QUICK_START.md" && echo "Deleted: tests/integration_tests/QUICK_START.md"
rm -f "tests/integration_tests/TEST_SUMMARY.md" && echo "Deleted: tests/integration_tests/TEST_SUMMARY.md"
rm -f "tests/performance/QUICK_START.md" && echo "Deleted: tests/performance/QUICK_START.md"

# website directory
rm -f "website/QUICKSTART.md" && echo "Deleted: website/QUICKSTART.md"
rm -f "website/docs/getting-started/quick-start.md" && echo "Deleted: website/docs/getting-started/quick-start.md"

echo ""
echo "Cleanup complete!"
echo "Note: The main docs/ files (docs/quickstart.md and docs/quick_reference.md) were preserved as they are part of the core documentation."
