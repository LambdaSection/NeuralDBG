# Cleanup Scripts Index

Quick reference to all cleanup utilities in the Neural DSL repository.

## Documentation Cleanup

**Purpose**: Remove 67+ redundant QUICK*.md and *SUMMARY.md files

### Scripts
- `delete_quick_and_summary_docs.bat` - Windows Command Prompt
- `delete_quick_and_summary_docs.sh` - Unix/Linux/macOS
- `delete_quick_and_summary_docs.ps1` - Windows PowerShell
- `cleanup_quick_and_summary_docs.py` - Python (cross-platform)

### Documentation
- **Main Guide**: `QUICK_SUMMARY_CLEANUP_README.md`
- **Implementation**: `DOCUMENTATION_CLEANUP_IMPLEMENTATION.md`
- **File Manifest**: `FILES_TO_DELETE_MANIFEST.txt`
- **Historical Record**: `docs/archive/QUICK_SUMMARY_CLEANUP_2025.md`

### Usage
```bash
# Windows
delete_quick_and_summary_docs.bat

# Unix/Linux/macOS
chmod +x delete_quick_and_summary_docs.sh
./delete_quick_and_summary_docs.sh

# PowerShell
.\delete_quick_and_summary_docs.ps1

# Python
python cleanup_quick_and_summary_docs.py
```

---

## Cache & Artifacts Cleanup

**Purpose**: Remove cache directories, virtual environments, and test artifacts

### Scripts
- `cleanup_cache_and_artifacts.bat` - Windows Command Prompt
- `cleanup_cache_and_artifacts.sh` - Unix/Linux/macOS
- `cleanup_cache_and_artifacts.ps1` - Windows PowerShell
- `cleanup_cache_and_artifacts.py` - Python (cross-platform)

### Documentation
- **Main Guide**: `CLEANUP_README.md`

### Usage
```bash
# Windows
cleanup_cache_and_artifacts.bat

# Unix/Linux/macOS
chmod +x cleanup_cache_and_artifacts.sh
./cleanup_cache_and_artifacts.sh

# PowerShell
.\cleanup_cache_and_artifacts.ps1

# Python
python cleanup_cache_and_artifacts.py
```

---

## Quick Reference

| Task | Command (Windows) | Command (Unix/Linux/macOS) |
|------|-------------------|----------------------------|
| Remove QUICK/SUMMARY docs | `delete_quick_and_summary_docs.bat` | `./delete_quick_and_summary_docs.sh` |
| Clean cache & artifacts | `cleanup_cache_and_artifacts.bat` | `./cleanup_cache_and_artifacts.sh` |
| Full cleanup | Run both scripts above | Run both scripts above |

---

## What Gets Cleaned

### Documentation Cleanup
- 14 root-level *SUMMARY.md files
- 2 .github/ summary files
- 7 docs/ redundant files
- 2 examples/ files
- 33 neural/ module quick-starts
- 1 scripts/ file
- 6 tests/ files
- 2 website/ files

**Total**: 67 files

### Cache & Artifacts Cleanup
- `__pycache__/` directories
- `.pytest_cache/` directories
- `.hypothesis/` directories
- `.mypy_cache/` directories
- `.ruff_cache/` directories
- `.venv/`, `venv/` directories
- `test_*.html`, `test_*.png` files
- `htmlcov/` directories
- `sample_*.py` temporary scripts

---

## Comprehensive Cleanup Workflow

For a complete repository cleanup:

### Step 1: Clean Documentation
```bash
# Choose your platform
delete_quick_and_summary_docs.bat   # Windows
./delete_quick_and_summary_docs.sh  # Unix/Linux/macOS
```

### Step 2: Clean Cache & Artifacts
```bash
# Choose your platform
cleanup_cache_and_artifacts.bat     # Windows
./cleanup_cache_and_artifacts.sh    # Unix/Linux/macOS
```

### Step 3: Verify
```bash
git status
```

### Step 4: Commit (if satisfied)
```bash
git add -A
git commit -m "chore: comprehensive repository cleanup"
```

### Step 5: Remove Cleanup Scripts (optional)
```bash
# Remove documentation cleanup scripts
rm delete_quick_and_summary_docs.{bat,sh,ps1}
rm cleanup_quick_and_summary_docs.py
rm QUICK_SUMMARY_CLEANUP_README.md
rm FILES_TO_DELETE_MANIFEST.txt
rm DOCUMENTATION_CLEANUP_IMPLEMENTATION.md
rm CLEANUP_SCRIPTS_INDEX.md

# Cache cleanup scripts should be kept for future use
```

---

## Documentation Structure

After cleanup, documentation is organized as:

```
Neural/
├── README.md                    # Project overview
├── AGENTS.md                    # Developer guide
├── CONTRIBUTING.md              # Contributing guidelines
├── CLEANUP_README.md            # Cleanup guide (keep)
├── docs/
│   ├── quickstart.md           # Main quickstart (preserved)
│   ├── quick_reference.md      # API reference (preserved)
│   ├── architecture/           # Architecture docs
│   ├── api/                    # API documentation
│   ├── tutorials/              # Tutorials
│   └── archive/                # Historical docs
│       └── QUICK_SUMMARY_CLEANUP_2025.md
└── [cleanup scripts]           # Optional to remove after use
```

---

## See Also

- **AGENTS.md** - Development guide with setup instructions
- **README.md** - Project overview and quick start
- **CLEANUP_README.md** - Detailed cleanup procedures
- **docs/quickstart.md** - Comprehensive quickstart guide
- **.gitignore** - Patterns for files to ignore (includes cache patterns)

---

**Last Updated**: January 2025
