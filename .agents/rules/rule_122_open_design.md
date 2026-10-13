---
name: rule-122-open-design
description: 'RULE 122: OpenDesign — donner le brief, pas de contrat forcé'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_122_open_design.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 122: OpenDesign — donner le brief, pas de contrat forcé
**Statut : SUGGESTED (défaut conseillé, jamais bloquant).**
Réf : https://github.com/nexu-io/open-design (alternative open-source à
Claude Design : l'agent devient le moteur de design).

## R122.1 — Principe
Pour toute tâche UI/design, le défaut conseillé est simple : **donner le
prompt (brief) à OpenDesign** au lieu de designer à la main.
- Via MCP : `od mcp install <agent>` (`<agent>` = opencode | codex | cursor
  | claude | copilot | …), puis `Use open-design to …` dans l'agent.
- Via skill directe si `od` indisponible : suivre `DESIGN.md` du projet
  quand il existe, sinon brief texte + capture de référence.
- Vérifier le rendu par screenshot avant de livrer (zéro console error).

## R122.2 — Pas de contrat forcé
- Aucun workflow obligatoire : ni `DESIGN.md` imposé, ni étape imposée.
- Si le projet a un `DESIGN.md`, le lire (source tokens/voix).
  Sinon : brief + screenshot suffisent.
- Complète R108/R109 (identité produit) sans les remplacer : en cas de
  conflit visuel, R108 gagne.

## R122.3 — Interdits (légers)
- Ne pas réinventer un design system quand OpenDesign ou R108 couvre le besoin.
- Ne pas livrer une UI non regardée (au minimum 1 screenshot relu).
