# Documentation Cleanup Scripts

This document describes the scripts used for cleaning up redundant documentation files.

## Overview

The documentation cleanup process removes 60+ redundant files and consolidates essential information into the main `docs/` directory.

## Scripts

### 1. Main Cleanup Script

**File**: `cleanup_redundant_docs.py`

**Purpose**: Removes all redundant documentation files

**Usage**:
```bash
python cleanup_redundant_docs.py
```

**What it does**:
- Removes 60+ redundant QUICK_*.md, *_SUMMARY.md, and implementation files
- Provides progress feedback
- Reports deleted and not-found files
- Lists preserved essential documentation

**Files removed**:
- QUICK_*.md files (except docs/quick_reference.md)
- *_SUMMARY.md files (except DOCUMENTATION_CLEANUP_SUMMARY.md)
- *_IMPLEMENTATION*.md files
- *_COMPLETE.md files
- Marketing automation documentation
- Obsolete cleanup and status files

### 2. Master Execution Script

**File**: `run_documentation_cleanup.py`

**Purpose**: Orchestrates the cleanup process with user confirmation

**Usage**:
```bash
python run_documentation_cleanup.py
```

**What it does**:
- Displays cleanup plan
- Asks for user confirmation
- Executes cleanup script
- Provides next steps

**Features**:
- Interactive confirmation
- Progress reporting
- Comprehensive output
- Next steps guidance

### 3. Verification Script

**File**: `verify_documentation_cleanup.py`

**Purpose**: Verifies cleanup results

**Usage**:
```bash
python verify_documentation_cleanup.py
```

**What it does**:
- Checks that redundant files were removed
- Verifies essential files are preserved
- Reports verification status
- Provides remediation guidance

**Checks**:
- Sample of removed files
- All essential preserved files
- Overall cleanup status

### 4. Listing Script

**File**: `list_remaining_docs.py`

**Purpose**: Lists remaining QUICK_*.md and *_SUMMARY.md files

**Usage**:
```bash
python list_remaining_docs.py
```

**What it does**:
- Finds all QUICK_*.md files
- Finds all *_SUMMARY.md files
- Finds all QUICKSTART*.md files
- Reports unexpected files
- Provides verification summary

**Output**:
- List of remaining quick reference files
- List of remaining summary files
- List of remaining quickstart files
- Verification status

## Workflow

### Recommended Execution Order

1. **Review the plan** (optional):
   ```bash
   python list_remaining_docs.py
   ```

2. **Execute cleanup** (with confirmation):
   ```bash
   python run_documentation_cleanup.py
   ```

3. **Verify results**:
   ```bash
   python verify_documentation_cleanup.py
   python list_remaining_docs.py
   ```

4. **Review changes**:
   ```bash
   git status
   git diff
   ```

5. **Commit changes**:
   ```bash
   git add -A
   git commit -m "docs: remove 60+ redundant documentation files"
   ```

### Direct Execution (No Confirmation)

If you want to run cleanup directly without confirmation:

```bash
python cleanup_redundant_docs.py
```

## Preserved Documentation

All essential documentation is preserved:

### Root Level
- `README.md` - Main project documentation
- `AGENTS.md` - Development guide
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `CLEANUP_README.md` - Cache cleanup guide
- `DOCUMENTATION_CLEANUP_SUMMARY.md` - Cleanup summary
- `DOCUMENTATION_CLEANUP_SCRIPTS.md` - This file
- Other essential files (SECURITY.md, LICENSE.md, etc.)

### Docs Directory
- `docs/README.md` - Documentation navigation
- `docs/quick_reference.md` - Consolidated quick reference ⭐
- `docs/FOCUS.md` - Project scope
- `docs/TYPE_SAFETY.md` - Type checking guidelines
- All core feature documentation
- All tutorial documentation
- All example documentation

## Rollback

All removed files are preserved in git history.

### View File History

```bash
# View deleted file from history
git log --all --full-history -- path/to/deleted/file.md
git show HEAD~1:path/to/deleted/file.md
```

### Restore a File

```bash
# Find the commit before deletion
git log --all --full-history -- path/to/deleted/file.md

# Restore the file
git checkout <commit-hash> -- path/to/deleted/file.md
```

### Undo Entire Cleanup

```bash
# Before committing
git reset --hard HEAD

# After committing
git revert <commit-hash>
```

## Troubleshooting

### Script Not Found

```bash
# Make sure you're in the repository root
cd /path/to/Neural

# Verify script exists
ls cleanup_redundant_docs.py
```

### Permission Denied (Unix/Linux/macOS)

```bash
# Make script executable
chmod +x cleanup_redundant_docs.py
chmod +x run_documentation_cleanup.py
```

### File Not Deleted

If a file wasn't deleted:
1. Check if the file path in the script is correct
2. Verify the file exists before cleanup
3. Check file permissions
4. Review error messages in script output

### Unexpected Files Remaining

If verification shows unexpected files:
1. Review the list of remaining files
2. Determine if they should be removed
3. Add them to the cleanup script if needed
4. Re-run cleanup

## Related Documentation

- [DOCUMENTATION_CLEANUP_SUMMARY.md](DOCUMENTATION_CLEANUP_SUMMARY.md) - Comprehensive cleanup summary
- [DOCUMENTATION_CLEANUP_README.md](DOCUMENTATION_CLEANUP_README.md) - Quick guide
- [docs/README.md](docs/README.md) - Documentation navigation
- [CHANGELOG.md](CHANGELOG.md) - Version history with cleanup notes

## Script Maintenance

### Adding Files to Cleanup

To add more files to the cleanup:

1. Edit `cleanup_redundant_docs.py`
2. Add file paths to `FILES_TO_DELETE` list
3. Test with dry-run if desired
4. Execute cleanup

### Updating Verification

To update verification checks:

1. Edit `verify_documentation_cleanup.py`
2. Update `FILES_TO_CHECK_REMOVED` list
3. Update `FILES_TO_CHECK_PRESERVED` list
4. Run verification

## Support

If you encounter issues:

1. Check this documentation
2. Review script output for error messages
3. Check git history for file locations
4. Open an issue on GitHub
5. Ask on Discord

---

**Last Updated**: January 2025  
**Scripts Version**: 1.0  
**Compatible With**: Neural DSL v0.4.0+
