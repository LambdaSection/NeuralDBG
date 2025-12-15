@echo off
REM Batch script to delete QUICK*.md and *SUMMARY.md documentation files
REM Run this script from the repository root

echo Deleting QUICK*.md and *SUMMARY.md files...
echo.

REM Root level files
if exist "BUG_FIXES_SUMMARY.md" del "BUG_FIXES_SUMMARY.md" && echo Deleted: BUG_FIXES_SUMMARY.md
if exist "CACHE_CLEANUP_SUMMARY.md" del "CACHE_CLEANUP_SUMMARY.md" && echo Deleted: CACHE_CLEANUP_SUMMARY.md
if exist "CLEANUP_SUMMARY.md" del "CLEANUP_SUMMARY.md" && echo Deleted: CLEANUP_SUMMARY.md
if exist "CLI_CLEANUP_SUMMARY.md" del "CLI_CLEANUP_SUMMARY.md" && echo Deleted: CLI_CLEANUP_SUMMARY.md
if exist "CONSOLIDATION_SUMMARY.md" del "CONSOLIDATION_SUMMARY.md" && echo Deleted: CONSOLIDATION_SUMMARY.md
if exist "DEPENDENCY_FIX_SUMMARY.md" del "DEPENDENCY_FIX_SUMMARY.md" && echo Deleted: DEPENDENCY_FIX_SUMMARY.md
if exist "DOCKER_CONSOLIDATION_SUMMARY.md" del "DOCKER_CONSOLIDATION_SUMMARY.md" && echo Deleted: DOCKER_CONSOLIDATION_SUMMARY.md
if exist "ERROR_SUGGESTIONS_FIX_SUMMARY.md" del "ERROR_SUGGESTIONS_FIX_SUMMARY.md" && echo Deleted: ERROR_SUGGESTIONS_FIX_SUMMARY.md
if exist "FINAL_FIXES_SUMMARY.md" del "FINAL_FIXES_SUMMARY.md" && echo Deleted: FINAL_FIXES_SUMMARY.md
if exist "HPO_FIXES_SUMMARY.md" del "HPO_FIXES_SUMMARY.md" && echo Deleted: HPO_FIXES_SUMMARY.md
if exist "IMPLEMENTATION_SUMMARY.md" del "IMPLEMENTATION_SUMMARY.md" && echo Deleted: IMPLEMENTATION_SUMMARY.md
if exist "INTEGRATIONS_SIMPLIFICATION_SUMMARY.md" del "INTEGRATIONS_SIMPLIFICATION_SUMMARY.md" && echo Deleted: INTEGRATIONS_SIMPLIFICATION_SUMMARY.md
if exist "LINTING_FIXES_SUMMARY.md" del "LINTING_FIXES_SUMMARY.md" && echo Deleted: LINTING_FIXES_SUMMARY.md
if exist "TEAMS_SIMPLIFICATION_SUMMARY.md" del "TEAMS_SIMPLIFICATION_SUMMARY.md" && echo Deleted: TEAMS_SIMPLIFICATION_SUMMARY.md

REM .github directory
if exist ".github\MARKETING_AUTOMATION_SUMMARY.md" del ".github\MARKETING_AUTOMATION_SUMMARY.md" && echo Deleted: .github\MARKETING_AUTOMATION_SUMMARY.md
if exist ".github\SECURITY_IMPLEMENTATION_SUMMARY.md" del ".github\SECURITY_IMPLEMENTATION_SUMMARY.md" && echo Deleted: .github\SECURITY_IMPLEMENTATION_SUMMARY.md

REM docs directory
if exist "docs\CONSOLIDATION_SUMMARY.md" del "docs\CONSOLIDATION_SUMMARY.md" && echo Deleted: docs\CONSOLIDATION_SUMMARY.md
if exist "docs\TRANSFORMER_DOCS_SUMMARY.md" del "docs\TRANSFORMER_DOCS_SUMMARY.md" && echo Deleted: docs\TRANSFORMER_DOCS_SUMMARY.md

REM docs/aquarium directory
if exist "docs\aquarium\DOCUMENTATION_CONSOLIDATION_SUMMARY.md" del "docs\aquarium\DOCUMENTATION_CONSOLIDATION_SUMMARY.md" && echo Deleted: docs\aquarium\DOCUMENTATION_CONSOLIDATION_SUMMARY.md
if exist "docs\aquarium\IMPLEMENTATION_SUMMARY.md" del "docs\aquarium\IMPLEMENTATION_SUMMARY.md" && echo Deleted: docs\aquarium\IMPLEMENTATION_SUMMARY.md
if exist "docs\aquarium\QUICK_REFERENCE.md" del "docs\aquarium\QUICK_REFERENCE.md" && echo Deleted: docs\aquarium\QUICK_REFERENCE.md

REM docs/mlops directory
if exist "docs\mlops\QUICK_REFERENCE.md" del "docs\mlops\QUICK_REFERENCE.md" && echo Deleted: docs\mlops\QUICK_REFERENCE.md

REM examples directory
if exist "examples\IMPLEMENTATION_SUMMARY.md" del "examples\IMPLEMENTATION_SUMMARY.md" && echo Deleted: examples\IMPLEMENTATION_SUMMARY.md
if exist "examples\attention_examples\QUICKSTART.md" del "examples\attention_examples\QUICKSTART.md" && echo Deleted: examples\attention_examples\QUICKSTART.md

REM neural/aquarium directory
if exist "neural\aquarium\EXPORT_IMPLEMENTATION_SUMMARY.md" del "neural\aquarium\EXPORT_IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\aquarium\EXPORT_IMPLEMENTATION_SUMMARY.md
if exist "neural\aquarium\IMPLEMENTATION_SUMMARY.md" del "neural\aquarium\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\aquarium\IMPLEMENTATION_SUMMARY.md
if exist "neural\aquarium\PACKAGING_SUMMARY.md" del "neural\aquarium\PACKAGING_SUMMARY.md" && echo Deleted: neural\aquarium\PACKAGING_SUMMARY.md
if exist "neural\aquarium\PLUGIN_IMPLEMENTATION_SUMMARY.md" del "neural\aquarium\PLUGIN_IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\aquarium\PLUGIN_IMPLEMENTATION_SUMMARY.md
if exist "neural\aquarium\PROJECT_SUMMARY.md" del "neural\aquarium\PROJECT_SUMMARY.md" && echo Deleted: neural\aquarium\PROJECT_SUMMARY.md
if exist "neural\aquarium\QUICK_REFERENCE.md" del "neural\aquarium\QUICK_REFERENCE.md" && echo Deleted: neural\aquarium\QUICK_REFERENCE.md
if exist "neural\aquarium\QUICK_START.md" del "neural\aquarium\QUICK_START.md" && echo Deleted: neural\aquarium\QUICK_START.md
if exist "neural\aquarium\QUICKSTART.md" del "neural\aquarium\QUICKSTART.md" && echo Deleted: neural\aquarium\QUICKSTART.md

REM neural/aquarium/src/components subdirectories
if exist "neural\aquarium\src\components\debugger\IMPLEMENTATION_SUMMARY.md" del "neural\aquarium\src\components\debugger\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\aquarium\src\components\debugger\IMPLEMENTATION_SUMMARY.md
if exist "neural\aquarium\src\components\editor\QUICKSTART.md" del "neural\aquarium\src\components\editor\QUICKSTART.md" && echo Deleted: neural\aquarium\src\components\editor\QUICKSTART.md
if exist "neural\aquarium\src\components\editor\SUMMARY.md" del "neural\aquarium\src\components\editor\SUMMARY.md" && echo Deleted: neural\aquarium\src\components\editor\SUMMARY.md
if exist "neural\aquarium\src\components\terminal\IMPLEMENTATION_SUMMARY.md" del "neural\aquarium\src\components\terminal\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\aquarium\src\components\terminal\IMPLEMENTATION_SUMMARY.md
if exist "neural\aquarium\src\components\terminal\QUICKSTART.md" del "neural\aquarium\src\components\terminal\QUICKSTART.md" && echo Deleted: neural\aquarium\src\components\terminal\QUICKSTART.md

REM neural subdirectories
if exist "neural\automl\QUICK_START.md" del "neural\automl\QUICK_START.md" && echo Deleted: neural\automl\QUICK_START.md
if exist "neural\config\QUICKSTART.md" del "neural\config\QUICKSTART.md" && echo Deleted: neural\config\QUICKSTART.md
if exist "neural\config\SUMMARY.md" del "neural\config\SUMMARY.md" && echo Deleted: neural\config\SUMMARY.md
if exist "neural\cost\QUICK_REFERENCE.md" del "neural\cost\QUICK_REFERENCE.md" && echo Deleted: neural\cost\QUICK_REFERENCE.md
if exist "neural\dashboard\QUICKSTART.md" del "neural\dashboard\QUICKSTART.md" && echo Deleted: neural\dashboard\QUICKSTART.md
if exist "neural\data\QUICKSTART.md" del "neural\data\QUICKSTART.md" && echo Deleted: neural\data\QUICKSTART.md
if exist "neural\education\IMPLEMENTATION_SUMMARY.md" del "neural\education\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\education\IMPLEMENTATION_SUMMARY.md
if exist "neural\education\QUICK_START.md" del "neural\education\QUICK_START.md" && echo Deleted: neural\education\QUICK_START.md
if exist "neural\integrations\QUICK_REFERENCE.md" del "neural\integrations\QUICK_REFERENCE.md" && echo Deleted: neural\integrations\QUICK_REFERENCE.md
if exist "neural\monitoring\IMPLEMENTATION_SUMMARY.md" del "neural\monitoring\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\monitoring\IMPLEMENTATION_SUMMARY.md
if exist "neural\monitoring\QUICKSTART.md" del "neural\monitoring\QUICKSTART.md" && echo Deleted: neural\monitoring\QUICKSTART.md
if exist "neural\no_code\QUICKSTART.md" del "neural\no_code\QUICKSTART.md" && echo Deleted: neural\no_code\QUICKSTART.md
if exist "neural\parser\REFACTORING_SUMMARY.md" del "neural\parser\REFACTORING_SUMMARY.md" && echo Deleted: neural\parser\REFACTORING_SUMMARY.md
if exist "neural\profiling\IMPLEMENTATION_SUMMARY.md" del "neural\profiling\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\profiling\IMPLEMENTATION_SUMMARY.md
if exist "neural\teams\QUICK_START.md" del "neural\teams\QUICK_START.md" && echo Deleted: neural\teams\QUICK_START.md
if exist "neural\tracking\IMPLEMENTATION_SUMMARY.md" del "neural\tracking\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\tracking\IMPLEMENTATION_SUMMARY.md
if exist "neural\tracking\QUICK_REFERENCE.md" del "neural\tracking\QUICK_REFERENCE.md" && echo Deleted: neural\tracking\QUICK_REFERENCE.md
if exist "neural\visualization\IMPLEMENTATION_SUMMARY.md" del "neural\visualization\IMPLEMENTATION_SUMMARY.md" && echo Deleted: neural\visualization\IMPLEMENTATION_SUMMARY.md
if exist "neural\visualization\QUICKSTART_GALLERY.md" del "neural\visualization\QUICKSTART_GALLERY.md" && echo Deleted: neural\visualization\QUICKSTART_GALLERY.md

REM scripts directory
if exist "scripts\automation\IMPLEMENTATION_SUMMARY.md" del "scripts\automation\IMPLEMENTATION_SUMMARY.md" && echo Deleted: scripts\automation\IMPLEMENTATION_SUMMARY.md

REM tests directory
if exist "tests\TEST_COVERAGE_SUMMARY.md" del "tests\TEST_COVERAGE_SUMMARY.md" && echo Deleted: tests\TEST_COVERAGE_SUMMARY.md
if exist "tests\benchmarks\QUICK_REFERENCE.md" del "tests\benchmarks\QUICK_REFERENCE.md" && echo Deleted: tests\benchmarks\QUICK_REFERENCE.md
if exist "tests\benchmarks\SUMMARY.md" del "tests\benchmarks\SUMMARY.md" && echo Deleted: tests\benchmarks\SUMMARY.md
if exist "tests\integration_tests\QUICK_START.md" del "tests\integration_tests\QUICK_START.md" && echo Deleted: tests\integration_tests\QUICK_START.md
if exist "tests\integration_tests\TEST_SUMMARY.md" del "tests\integration_tests\TEST_SUMMARY.md" && echo Deleted: tests\integration_tests\TEST_SUMMARY.md
if exist "tests\performance\QUICK_START.md" del "tests\performance\QUICK_START.md" && echo Deleted: tests\performance\QUICK_START.md

REM website directory
if exist "website\QUICKSTART.md" del "website\QUICKSTART.md" && echo Deleted: website\QUICKSTART.md
if exist "website\docs\getting-started\quick-start.md" del "website\docs\getting-started\quick-start.md" && echo Deleted: website\docs\getting-started\quick-start.md

echo.
echo Cleanup complete!
echo Note: The main docs/ files (docs\quickstart.md and docs\quick_reference.md) were preserved as they are part of the core documentation.
