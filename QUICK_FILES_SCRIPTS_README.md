# Quick Reference Files - Consolidation Scripts

This directory contains scripts for managing the consolidation of QUICK*.md files.

## Scripts

### 1. `consolidate_quick_refs.py`
**Purpose**: Main consolidation script (already executed)

Performs:
- Finds all QUICK*.md files
- Creates consolidated `docs/quick_reference.md`
- Replaces QUICK files with deprecation notices

**Status**: ✅ Already executed

### 2. `remove_quick_files.py`
**Purpose**: Simple file removal script

Contains the list of all QUICK*.md files to deprecate. Alternative to `consolidate_quick_refs.py`.

**Status**: ✅ Already executed (files deprecated)

### 3. `cleanup_deprecated_quick_files.py`
**Purpose**: Remove deprecated files after transition period

**When to use**: After 3-6 months transition period

**Usage**:
```bash
# Review what will be deleted
python cleanup_deprecated_quick_files.py

# Follow prompts to confirm deletion
```

**Features**:
- Finds all files marked as DEPRECATED
- Shows list of files to be deleted
- Requires confirmation before deletion
- Preserves important files (quick_reference.md, etc.)
- Provides git commit instructions

**Important**: Only run this after users have had sufficient time to transition!

## Files Created by Consolidation

1. **docs/quick_reference.md** - The new consolidated quick reference
2. **QUICK_FILES_CONSOLIDATION.md** - Detailed consolidation report
3. **CONSOLIDATION_SUMMARY.md** - Executive summary
4. **docs/archive/QUICK_FILES_CLEANUP_2025.md** - Archive note

## Current State

✅ **Completed Steps**:
- All 43+ QUICK*.md files deprecated
- New consolidated guide created
- Documentation updated with new references
- Deprecation notices in place

📋 **Future Steps** (Optional):
- Wait 3-6 months for user transition
- Run `cleanup_deprecated_quick_files.py` to permanently remove files
- Update any external documentation
- Archive any remaining unique content

## Verification

Check consolidation status:

```bash
# Find all QUICK*.md files
find . -name "*QUICK*.md" -type f

# Count deprecated files
grep -r "DEPRECATED" **/*QUICK*.md | wc -l

# Verify new guide exists
cat docs/quick_reference.md
```

## Rollback

If needed, restore original files from git:

```bash
# View history of a specific file
git log -- neural/dashboard/QUICKSTART.md

# Restore a file to its previous state
git checkout <commit-hash> -- neural/dashboard/QUICKSTART.md
```

## Notes

- All deprecated files redirect users to `docs/quick_reference.md`
- No essential information was lost in consolidation
- Scripts are safe to re-run (idempotent)
- Git history preserves all original content

## Support

For questions about the consolidation:
1. See `QUICK_FILES_CONSOLIDATION.md` for detailed report
2. See `CONSOLIDATION_SUMMARY.md` for executive summary
3. Check `docs/quick_reference.md` for the new guide
4. Review git history for original content
