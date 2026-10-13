---
name: rule-116-regression-guards
description: 'RULE 116: Regression Guards — Les Erreurs Corrigées Ne Reviennent Pas'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_116_regression_guards.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 116: Regression Guards — Les Erreurs Corrigées Ne Reviennent Pas

## Problem

Session 2026-09-21 (pipeline X + Discord + Epingle) : 5 classes d'erreurs auraient
dû être impossibles avec des garde-fous :

1. **Parser aveugle** : `parse_epingle` ignorait `## Laboratoire` (Horcruxe Labs
   invisible) ; le registre ownership ignorait 7 orgs λ (OpenQuant exclu).
2. **Projets fantômes** : 14 entrées `projects.txt` absentes d'Epingle (R85),
   dont Forma à 25 commits/30j.
3. **Alias non résolu** : dossier `kuro` vs ligne Epingle `KuroGuardian` —
   vélocité jamais calculée des deux côtés.
4. **Artefacts périmés** : draft du jour hors-sélection laissé dans `outputs/`.
5. **Formats partagés cassés** : restructuration JSON sans vérifier les
   consommateurs (`kuro_investor_digest.py`).

## Solution

- `scripts/check_epingle_completeness.py` — gate R85 : ERROR si projet suivi +
  remote OWNED vérifié + absent d'Epingle ; WARN pour forks, orgs inconnues,
  sans-remote (décision humaine R87). Registre unique via
  `gen_x_posts.OWNED_MARKERS` / `classify_ownership`.
- `gen_x_posts.REPO_ALIASES` + `is_tracked()` — un dossier = une ligne Epingle,
  jamais de doublon (R80), jamais d'oubli (R85).
- `write_drafts()` nettoie les drafts du jour hors-sélection (même date uniquement).
- Avant de modifier un format partagé : `grep` des consommateurs + test du
  consommateur existant.
- Pas de Python inline complexe via PowerShell (quoting) : fichier script + `pytest`.

## Verification

```
python scripts/check_epingle_completeness.py        # exit 1 si ERROR (CI : gate)
python scripts/gen_x_posts.py --dry-run --top 3     # picks plausibles vs TRUTH_DAILY
python -m pytest tests/ -q                          # 100% vert avant de clore
```

Branchement CI (`.github/workflows/kuro.yml`, après l'audit truth) :
```
python kuro-rules/scripts/check_epingle_completeness.py --warn-only
```
`--warn-only` en CI (le robot ne casse jamais sur un WARN) ; strict en local.

---

**Créée** : 2026-09-21
**Déclencheur** : Session pipeline X/Discord — 14 projets réintégrés, orgs satellites, alias kuro
**Application** : MANDATORY — tout ajout parser/registre/format partagé
**Associée** : R80, R85, R87, R89, R93, R94-v2, R99, R105
