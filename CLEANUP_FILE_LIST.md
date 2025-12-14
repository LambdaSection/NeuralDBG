# Cleanup File List

This document lists all files that will be archived or removed during the repository cleanup.

## Files to Archive (50+ files → docs/archive/)

These implementation summaries and internal tracking documents will be moved to `docs/archive/`:

### Implementation Summaries
1. AQUARIUM_IMPLEMENTATION_SUMMARY.md
2. BENCHMARKS_IMPLEMENTATION_SUMMARY.md
3. COST_OPTIMIZATION_IMPLEMENTATION.md
4. DATA_VERSIONING_IMPLEMENTATION.md
5. IMPLEMENTATION_CHECKLIST.md
6. IMPLEMENTATION_COMPLETE.md
7. IMPLEMENTATION_SUMMARY.md
8. INTEGRATION_IMPLEMENTATION.md
9. INTEGRATIONS_SUMMARY.md
10. MARKETPLACE_IMPLEMENTATION.md
11. MARKETPLACE_SUMMARY.md
12. MLOPS_IMPLEMENTATION.md
13. MULTIHEADATTENTION_IMPLEMENTATION.md
14. NEURAL_API_IMPLEMENTATION.md
15. PERFORMANCE_IMPLEMENTATION.md
16. POSITIONAL_ENCODING_IMPLEMENTATION.md
17. POST_RELEASE_IMPLEMENTATION_SUMMARY.md
18. TEAMS_IMPLEMENTATION.md
19. TRANSFORMER_DECODER_IMPLEMENTATION.md
20. WEBSITE_IMPLEMENTATION_SUMMARY.md

### Feature Enhancements and Quick References
21. CLOUD_IMPROVEMENTS_SUMMARY.md
22. DEPENDENCY_OPTIMIZATION_SUMMARY.md
23. DEPLOYMENT_FEATURES.md
24. DOCUMENTATION_SUMMARY.md
25. POST_RELEASE_AUTOMATION_QUICK_REF.md
26. QUICK_START_AUTOMATION.md
27. TRANSFORMER_ENHANCEMENTS.md
28. TRANSFORMER_QUICK_REFERENCE.md

### Release and Version Documentation
29. GITHUB_PUBLISHING_GUIDE.md
30. GITHUB_RELEASE_v0.3.0.md
31. RELEASE_NOTES_v0.3.0.md
32. RELEASE_VERIFICATION_v0.3.0.md
33. V0.3.0_RELEASE_SUMMARY.md

### Migration and Setup Guides
34. MIGRATION_GUIDE_DEPENDENCIES.md
35. MIGRATION_v0.3.0.md
36. SETUP_STATUS.md

### Distribution and Tracking
37. DISTRIBUTION_JOURNAL.md
38. DISTRIBUTION_PLAN.md
39. DISTRIBUTION_QUICK_REF.md

### Miscellaneous Internal Documentation
40. AUTOMATION_GUIDE.md
41. BUG_FIXES.md
42. CHANGES_SUMMARY.md
43. CLEANUP_PLAN.md
44. DEPENDENCY_CHANGES.md
45. ERROR_MESSAGES_GUIDE.md
46. EXTRACTED_PROJECTS.md
47. IMPORT_REFACTOR.md
48. REPOSITORY_STRUCTURE.md
49. WEBSITE_README.md

## Files to Delete Completely

These obsolete development scripts will be removed:

1. repro_parser.py - Temporary reproduction script
2. reproduce_issue.py - Temporary debug script
3. _install_dev.py - Obsolete installer
4. _setup_repo.py - Obsolete setup script
5. install.bat - Obsolete Windows installer
6. install_dev.bat - Obsolete Windows dev installer
7. install_deps.py - Obsolete dependency installer

## GitHub Actions Workflows to Remove (16 files)

Replaced by 4 consolidated workflows:

1. .github/workflows/aquarium-release.yml
2. .github/workflows/automated_release.yml
3. .github/workflows/benchmarks.yml
4. .github/workflows/ci.yml *(replaced by essential-ci.yml)*
5. .github/workflows/close-fixed-issues.yml
6. .github/workflows/complexity.yml
7. .github/workflows/metrics.yml
8. .github/workflows/periodic_tasks.yml
9. .github/workflows/post_release.yml
10. .github/workflows/pre-commit.yml
11. .github/workflows/pylint.yml *(consolidated into essential-ci.yml)*
12. .github/workflows/pypi.yml *(replaced by release.yml)*
13. .github/workflows/pytest-to-issues.yml
14. .github/workflows/python-publish.yml *(replaced by release.yml)*
15. .github/workflows/security-audit.yml *(consolidated into essential-ci.yml)*
16. .github/workflows/security.yml *(consolidated into essential-ci.yml)*
17. .github/workflows/snyk-security.yml
18. .github/workflows/validate_examples.yml *(replaced by validate-examples.yml)*

## Files to Keep (Essential Documentation)

These essential files remain in the root directory:

### Core Documentation
- README.md - Project overview and getting started
- CHANGELOG.md - Version history
- CONTRIBUTING.md - Contribution guidelines
- LICENSE.md - Project license

### Setup and Installation
- INSTALL.md - Installation instructions
- GETTING_STARTED.md - Quick start guide
- AGENTS.md - Agent/automation guide
- SECURITY.md - Security policy

### Dependencies
- DEPENDENCY_GUIDE.md - Dependency management guide
- DEPENDENCY_QUICK_REF.md - Quick reference for dependencies

### Configuration Files
- .gitignore - Git ignore patterns
- .pre-commit-config.yaml - Pre-commit hooks
- .pylintrc - Pylint configuration
- .prospector.yaml - Prospector configuration
- pyproject.toml - Project configuration
- setup.py - Setup script
- requirements*.txt - Dependency files
- mypy.ini - Mypy configuration
- codecov.yml - Codecov configuration
- neural-schema.json - DSL schema

### Docker
- Dockerfile
- docker-compose.yml
- docker-compose.prod.yml
- nginx.conf
- .dockerignore

### New Cleanup Files
- cleanup_redundant_files.py - Documentation cleanup script
- cleanup_workflows.py - Workflow cleanup script
- run_cleanup.py - Master cleanup script
- CLEANUP_EXECUTION.md - Execution guide
- REPOSITORY_CLEANUP_SUMMARY.md - Cleanup summary
- QUICK_REFERENCE.md - Quick reference card
- CLEANUP_FILE_LIST.md - This file

## New Essential Workflows

These new consolidated workflows replace the 18+ old workflows:

1. .github/workflows/essential-ci.yml - Main CI/CD pipeline
2. .github/workflows/release.yml - Unified release process
3. .github/workflows/codeql.yml - Security analysis
4. .github/workflows/validate-examples.yml - Example validation

## Summary Statistics

- **Documentation archived**: 49 files
- **Obsolete scripts deleted**: 7 files
- **Workflows removed**: 18 files
- **Workflows consolidated to**: 4 files
- **Essential docs kept**: 13 files
- **Total files cleaned**: 74 files

## Impact

### Before Cleanup
- Root directory: 56+ markdown files
- GitHub workflows: 20 files
- Many redundant/obsolete files

### After Cleanup
- Root directory: 13 essential markdown files + cleanup docs
- GitHub workflows: 4 essential files
- Clean, focused repository structure

## Rollback Instructions

If you need to undo the cleanup:

```bash
# Restore archived documentation
mv docs/archive/*.md ./

# Restore workflows (from git history)
git checkout HEAD~1 .github/workflows/
```

All archived files are preserved and can be restored if needed.
