# Quick Reference Files Cleanup - December 2025

## Context

As part of the v0.3.0 repository cleanup, 43+ QUICK*.md files were deprecated and consolidated into a single authoritative quick reference guide.

## Action Taken

**Date**: December 2025

All QUICK*.md files across the repository were:
1. Replaced with deprecation notices
2. Redirected to `docs/quick_reference.md`
3. Scheduled for eventual removal

## Consolidated Information

All essential quick-start information is now in:
- **Primary**: `docs/quick_reference.md`
- **Secondary**: Main `README.md` (Getting Started section)

## Files Affected

See `QUICK_FILES_CONSOLIDATION.md` in the repository root for complete list of affected files.

## Future Cleanup

After a transition period (suggested: 3-6 months), the deprecated QUICK*.md files can be fully removed from the repository.

To remove deprecated files:

```bash
# List all deprecated QUICK files
find . -name "*QUICK*.md" -type f -exec grep -l "DEPRECATED" {} \;

# Review and remove
git rm <files>
git commit -m "Remove deprecated QUICK reference files"
```

## Archived Original Content

Key content from original files before consolidation:

### docs/quickstart.md
- Comprehensive quick-start guide (→ now in docs/quick_reference.md)
- Installation instructions
- First model creation
- Common commands

### neural/dashboard/QUICKSTART.md
- NeuralDbg dashboard setup (→ now in docs/quick_reference.md)
- API examples
- Profiler usage

### QUICK_FIXES.md
- Test fix guide (→ deprecated, test-specific)
- Import error fixes
- Parser fixes

Most other QUICK files were already deprecated or contained minimal unique content.

## References

- Main consolidation doc: `/QUICK_FILES_CONSOLIDATION.md`
- New quick reference: `/docs/quick_reference.md`
- Updated docs index: `/docs/README.md`
