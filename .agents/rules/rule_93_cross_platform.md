---
name: rule-93-cross-platform
description: 'RULE 93: Cross-Platform Reliability (Windows/Linux)'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_93_cross_platform.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 93: Cross-Platform Reliability (Windows/Linux)

## Rule

All infrastructure scripts, automation tools (Makefiles, Bash scripts), and core logic MUST be platform-agnostic or provide explicit support for both Windows and Linux environments.

## Rationale

NeuralDBG and related projects are developed and used across different operating systems. Brittle, Linux-only scripts (using `/tmp/`, `sed -i`, or `bin/` paths) break the workflow for Windows users and compromise the "Extreme Rigor" standard.

## Implementation Standards

1. **Path Agnosticism**: Use Python's `pathlib` or platform detection for directory paths.
    - Windows: `.venv/Scripts/python.exe`
    - Linux: `.venv/bin/python`
2. **Command Agnosticism**: Avoid platform-specific binaries in automation.
    - Use `python -c "..."` instead of `sed` or `awk`.
    - Use `python -m ...` instead of direct binary calls when possible.
3. **Environment Variables**: Use `$TEMP` or `tempfile` module instead of hardcoded `/tmp/`.
4. **Shell Selection**: If using `Makefile`, ensure it doesn't rely on specific Bash extensions that fail on Windows `cmd` or `powershell` unless wrapped.
5. **Terminal Output Encoding Protection**: Avoid using raw Unicode emojis (e.g., 🔍, 📊, ✅) in user-facing logging/console print outputs. On Windows command consoles utilizing local character maps (like CP1252), this causes a fatal `UnicodeEncodeError`. Use ASCII indicators instead (e.g., `[Search]`, `[Stats]`, `[OK]`).

## Verification

```
WHEN creating/modifying a script:
  1. Does it use hardcoded Linux-style paths (/usr, /tmp, bin/)?
  2. Does it use 'sed -i' or other non-native Windows commands?
  3. Have we tested it on the current USER environment (Windows)?
```

## Enforcement

```
IF script failure occurs due to OS mismatch:
  ACTION: Refactor to be platform-agnostic immediately.
  ACTION: Add detection logic to infrastructure entry points (e.g. Makefile).
```

---

**Created**: 2026-05-11
**Trigger**: Infrastructure failure on Windows environment
**Enforcement**: MANDATORY
