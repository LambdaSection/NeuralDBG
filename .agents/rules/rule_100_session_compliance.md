---
name: rule-100-session-compliance
description: Session compliance verification
trigger: always
scope: company
priority: 10
status: active
---

<!-- Generated from rules/rule_100_session_compliance.md — DO NOT EDIT. scope=company priority=10 status=active -->

# RULE 100: Session Compliance — Vérification Obligatoire en Début de Session

## Trigger
Au TOUT DÉBUT de chaque session de travail, avant toute autre action.

## Obligation
L'agent DOIT :
1. Lire `rules/COMPLIANCE_CHECKLIST.md`.
2. S'il existe un dossier `docs/tracking/` ou un fichier `docs/launch_plan_*.md`, lire obligatoirement `docs/tracking/acquisition_tracker.md` et le plan de lancement en début de session afin de synchroniser sa mémoire sur les publications marketing et l'historique d'acquisition.
3. Vérifier TOUTES les règles 🔴🔴 MANDATORY.
4. Signaler dans le premier message de la session l'état de conformité.
5. Si une règle mandatory est en 🔴 VIOLATED, NE PAS commencer à travailler avant correction.

## Format du message de début de session
```
📋 Compliance Check — {DATE}
🔴🔴 Mandatory: X/16 ✅ | ⏳ Y pending | 🔴 Z violated
🔵 Optional: X/XX ✅
⚠️ Violations: (liste si existe)
```

## Sanction
Si un agent commence à travailler sans avoir vérifié la compliance checklist :
- La session est marquée comme VIOLATED dans SESSION_SUMMARY.md
- L'agent DOIT documenter pourquoi la vérification a été sautée

## Fichiers associés
- `rules/COMPLIANCE_CHECKLIST.md` — la checklist à vérifier
- `SESSION_SUMMARY.md` — où logger les violations
