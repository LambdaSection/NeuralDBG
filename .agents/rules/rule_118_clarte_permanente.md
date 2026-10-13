---
name: rule-118-clarte-permanente
description: 'RULE 118: Clarté Permanente — Toujours Comme La Première Fois'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_118_clarte_permanente.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 118: Clarté Permanente — Toujours Comme La Première Fois

## Problem
L'utilisateur ne comprend pas toujours tout. Il demande : expliquer toujours clairement comme la première fois, partout, étape par étape, sans jargon non défini. Sans règle, l'IA redevient technique après 2-3 messages.

## Solution
Étend R65 + R72b. S'applique à CHAQUE réponse, pas seulement au code.

1. Étape par étape obligatoire : Étape 0, 1, 2... Jamais de bloc unique. Une idée = une étape.
2. Chaque mot technique défini à sa première apparition dans la réponse : Curry-Howard, Lean, MSR, graphe, spec, etc. Même si déjà expliqué avant. Répéter court.
3. Exemple concret Python à chaque concept abstrait. Pas de LaTeX. Pas de formule sans exemple.
4. Répondre d'abord à "c'est quoi ? un papier ? un outil ? une idée ?" avant de donner des détails.
5. Finir par "prochaine étape" en 1 phrase + question oui/non.
6. Langue de l'utilisateur en premier (ici français).

## Verification
- [ ] Chaque réponse contient des étapes numérotées ?
- [ ] Chaque terme technique a sa définition courte dans la même réponse ?
- [ ] Au moins 1 exemple concret par concept ?
- [ ] Si violation : réécrire la réponse avant d'envoyer, pas après.

## Enforcement
IF réponse sans étapes OU jargon non défini :
  ACTION: STOP — réécrire avec étapes + définitions + exemple
  DO NOT: renvoyer vers une explication passée ("comme dit plus haut")

Created: 2026-09-22 — demande fondatrice Horcruxe Labs 004 : "explique toujours tout clairement comme la première fois, je veux toujours ça partout"
Enforcement: MANDATORY pour ce workspace
