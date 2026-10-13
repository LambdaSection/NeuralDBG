---
name: rule-117-desktop-build
description: 'RULE 117: Desktop Build — Rebuild + Réinstall auto après chaque modif
  — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_117_desktop_build.md — DO NOT EDIT. scope=company priority=2 status=active -->

## RULE 117: Desktop Build — Rebuild + Réinstall auto après chaque modif — MANDATORY

### Rule

Toute modification du code d'une desktop app livrée (ex : `Horcruxe II/app.py`)
DOIT être suivie, dans le même tour, de :

1. **Selftest** : `python app.py --selftest` → `SELFTEST_OK`
2. **Rebuild** : `python -m PyInstaller --noconfirm --onefile --windowed --name <Nom> app.py`
3. **Réinstall** : `install.ps1` (copie vers `%LOCALAPPDATA%` + raccourcis)
4. **Vérif exe** : `<Nom>.exe --selftest` → exit 0, et processus GUI up si lancé

Jamais de code livré sans binaire à jour. Le `.exe` dans `dist/` et la version
installée doivent toujours correspondre au `app.py` courant.

### Verification

```
ACTION: après chaque edit de app.py :
VERIFY: 1. py_compile / --selftest OK
        2. dist/<Nom>.exe reconstruit (timestamp > edit)
        3. install.ps1 rejoué → INSTALL_OK
        4. exe --selftest → exit 0
```

### Enforcement

IF un edit de desktop app est fait sans rebuild dans le même tour :

- STOP, builder + réinstaller immédiatement, puis annoncer version + chemin exe

IF l'utilisateur dit "mets à jour" / "installe" :

- rebuild + install.ps1 + vérif processus, sans demander
