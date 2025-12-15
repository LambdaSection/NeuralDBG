# PowerShell script to delete QUICK*.md and *SUMMARY.md documentation files
# Run this script from the repository root

Write-Host "Deleting QUICK*.md and *SUMMARY.md files..." -ForegroundColor Cyan
Write-Host ""

# Root level files
Remove-Item -Path "BUG_FIXES_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: BUG_FIXES_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "CACHE_CLEANUP_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: CACHE_CLEANUP_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "CLEANUP_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: CLEANUP_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "CLI_CLEANUP_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: CLI_CLEANUP_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "CONSOLIDATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: CONSOLIDATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "DEPENDENCY_FIX_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: DEPENDENCY_FIX_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "DOCKER_CONSOLIDATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: DOCKER_CONSOLIDATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "ERROR_SUGGESTIONS_FIX_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: ERROR_SUGGESTIONS_FIX_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "FINAL_FIXES_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: FINAL_FIXES_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "HPO_FIXES_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: HPO_FIXES_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "INTEGRATIONS_SIMPLIFICATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: INTEGRATIONS_SIMPLIFICATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "LINTING_FIXES_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: LINTING_FIXES_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "TEAMS_SIMPLIFICATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: TEAMS_SIMPLIFICATION_SUMMARY.md" -ForegroundColor Green }

# .github directory
Remove-Item -Path ".github\MARKETING_AUTOMATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: .github\MARKETING_AUTOMATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path ".github\SECURITY_IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: .github\SECURITY_IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

# docs directory
Remove-Item -Path "docs\CONSOLIDATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: docs\CONSOLIDATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "docs\TRANSFORMER_DOCS_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: docs\TRANSFORMER_DOCS_SUMMARY.md" -ForegroundColor Green }

# docs/aquarium directory
Remove-Item -Path "docs\aquarium\DOCUMENTATION_CONSOLIDATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: docs\aquarium\DOCUMENTATION_CONSOLIDATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "docs\aquarium\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: docs\aquarium\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "docs\aquarium\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: docs\aquarium\QUICK_REFERENCE.md" -ForegroundColor Green }

# docs/mlops directory
Remove-Item -Path "docs\mlops\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: docs\mlops\QUICK_REFERENCE.md" -ForegroundColor Green }

# examples directory
Remove-Item -Path "examples\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: examples\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "examples\attention_examples\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: examples\attention_examples\QUICKSTART.md" -ForegroundColor Green }

# neural/aquarium directory
Remove-Item -Path "neural\aquarium\EXPORT_IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\EXPORT_IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\PACKAGING_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\PACKAGING_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\PLUGIN_IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\PLUGIN_IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\PROJECT_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\PROJECT_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\QUICK_REFERENCE.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\QUICK_START.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\QUICK_START.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\QUICKSTART.md" -ForegroundColor Green }

# neural/aquarium/src/components subdirectories
Remove-Item -Path "neural\aquarium\src\components\debugger\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\src\components\debugger\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\src\components\editor\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\src\components\editor\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\src\components\editor\SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\src\components\editor\SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\src\components\terminal\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\src\components\terminal\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\aquarium\src\components\terminal\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\aquarium\src\components\terminal\QUICKSTART.md" -ForegroundColor Green }

# neural subdirectories
Remove-Item -Path "neural\automl\QUICK_START.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\automl\QUICK_START.md" -ForegroundColor Green }

Remove-Item -Path "neural\config\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\config\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "neural\config\SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\config\SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\cost\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\cost\QUICK_REFERENCE.md" -ForegroundColor Green }

Remove-Item -Path "neural\dashboard\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\dashboard\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "neural\data\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\data\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "neural\education\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\education\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\education\QUICK_START.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\education\QUICK_START.md" -ForegroundColor Green }

Remove-Item -Path "neural\integrations\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\integrations\QUICK_REFERENCE.md" -ForegroundColor Green }

Remove-Item -Path "neural\monitoring\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\monitoring\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\monitoring\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\monitoring\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "neural\no_code\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\no_code\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "neural\parser\REFACTORING_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\parser\REFACTORING_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\profiling\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\profiling\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\teams\QUICK_START.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\teams\QUICK_START.md" -ForegroundColor Green }

Remove-Item -Path "neural\tracking\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\tracking\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\tracking\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\tracking\QUICK_REFERENCE.md" -ForegroundColor Green }

Remove-Item -Path "neural\visualization\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\visualization\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "neural\visualization\QUICKSTART_GALLERY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: neural\visualization\QUICKSTART_GALLERY.md" -ForegroundColor Green }

# scripts directory
Remove-Item -Path "scripts\automation\IMPLEMENTATION_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: scripts\automation\IMPLEMENTATION_SUMMARY.md" -ForegroundColor Green }

# tests directory
Remove-Item -Path "tests\TEST_COVERAGE_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: tests\TEST_COVERAGE_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "tests\benchmarks\QUICK_REFERENCE.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: tests\benchmarks\QUICK_REFERENCE.md" -ForegroundColor Green }

Remove-Item -Path "tests\benchmarks\SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: tests\benchmarks\SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "tests\integration_tests\QUICK_START.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: tests\integration_tests\QUICK_START.md" -ForegroundColor Green }

Remove-Item -Path "tests\integration_tests\TEST_SUMMARY.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: tests\integration_tests\TEST_SUMMARY.md" -ForegroundColor Green }

Remove-Item -Path "tests\performance\QUICK_START.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: tests\performance\QUICK_START.md" -ForegroundColor Green }

# website directory
Remove-Item -Path "website\QUICKSTART.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: website\QUICKSTART.md" -ForegroundColor Green }

Remove-Item -Path "website\docs\getting-started\quick-start.md" -ErrorAction SilentlyContinue
if ($?) { Write-Host "Deleted: website\docs\getting-started\quick-start.md" -ForegroundColor Green }

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Cyan
Write-Host "Note: The main docs/ files (docs\quickstart.md and docs\quick_reference.md) were preserved as they are part of the core documentation." -ForegroundColor Yellow
