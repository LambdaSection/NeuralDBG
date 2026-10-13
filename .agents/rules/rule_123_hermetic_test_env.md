---
name: rule-123-hermetic-test-env
description: 'RULE 123: Hermetic Tests & Demo Isolation — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_123_hermetic_test_env.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 123: Hermetic Tests & Demo Isolation — MANDATORY

## Problem

S1-S3 (Kuro x Helium, 2026-10-04) proved that tests can pass while
polluting the real node: on Windows, `dirs::home_dir()` uses WinAPI
(`SHGetKnownFolderPath`), NOT the process env — so `HOME`/`USERPROFILE`
overrides are silent no-ops and every "isolated" test wrote to the real
`~/.helium` (market.db, workloads.json). Same class of bug: demos writing
to real state, WSL interop dropping custom env for Windows .exe.

## Rule

1. **Explicit home override in code.** Any app resolving a state dir MUST
   honor an explicit env override checked BEFORE the OS lookup
   (ex: `HELIUM_HOME` absolute path wins over `dirs::home_dir()`).
   No override = production path, never test magic inside prod code.
2. **Env-mutating tests are serial.** Process env is shared by all test
   threads: guard every env-mutating test with a process-global static
   Mutex (poison-tolerant), restore env before asserting.
3. **Hermetic demos.** Demo scripts MUST export the override to a `mktemp`
   dir and touch nothing real. Document the working shell: on Windows run
   under Git-Bash/PowerShell — WSL interop does not forward custom env to
   Windows .exe (verified 2026-10-04).
4. **Prove isolation, don't assume it.** At least one test must fail if the
   override is ignored (ex: quota counting against a known-real state, or
   assert the temp dir received the writes).

## Verification

```
Before any test touching disk/net/process:
  IF no override respected by the code: STOP -> add HELIUM_HOME-style knob
  IF test mutates env without a serial lock: STOP -> add static Mutex
  IF demo writes outside mktemp: STOP -> export override first
After a test session on a dev machine:
  CHECK real state dirs (~/.helium, ~/.kuro) for test-named rows
  IF polluted: STOP -> clean + backup before continuing
```

## Violation (real, 2026-10-04)

**VIOLATION**: 3 HTTP tests overriding HOME/USERPROFILE, green locally,
writing kuro:test/kuro:s3 rows into the real market.db.
**CORRECT**: `HELIUM_HOME` in identity/market/workload/storage + serial
Mutex + temp dirs; real state cleaned with .bak backups.

---

**Created**: 2026-10-04
**Trigger**: S3 test pollution of real ~/.helium (R89 lesson)
**Applies to**: All Rust (dirs) and Python (Path.home, sockets, subprocess) projects
**Enforcement**: MANDATORY
**Pairs with**: R18 (full suite after changes), R89 (lessons as rules), R93 (cross-platform)
