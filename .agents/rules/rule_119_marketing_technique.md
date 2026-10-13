---
name: rule-119-marketing-technique
description: 'Technical marketing: help in public, 1-2 threads/week'
trigger: manual
scope: team
priority: 1
status: active
paths:
- docs/**
---

<!-- Generated from rules/rule_119_marketing_technique.md — DO NOT EDIT. scope=team priority=1 status=active -->

# RULE 119: Marketing Technique — Résoudre des Problèmes Réels en Public

**Principe** : la présentation attire l'attention, la résolution de problèmes
réels construit la confiance. Les deux tournent **en parallèle**, jamais l'un
sans l'autre. Le marketing technique = aider publiquement là où l'audience
souffre déjà, avec le produit en preuve — pas en slogan.

## Les deux pistes (R116 + R119)

| | Présentation (R116) | Technique (R119, cette règle) |
|---|---|---|
| Quoi | Posts, lancements, démos | Réponses à des problèmes ouverts |
| Où | X, Discord, Reddit, HN | GitHub issues, forums, Stack Overflow |
| Métrique | Vues, stars | Réponses reçues, conversions pilote |
| Risque | Bruit, indifférence | Spam perçu (voir garde-fous) |

## Processus hebdomadaire

1. **Veille** (automatisé) : `gh search issues` + agent-reach sur les
   mots du domaine (ex : P2P, discovery LAN, shared storage, GPU).
2. **Qualification** (tout DOIT être vrai) :
   - Issue ouverte, 0 à 2 réponses existantes
   - Problème qu'on a VRAIMENT résolu (preuve : logs, code, benchmarks)
   - Moins de 6 mois d'inactivité (pas de nécro sur threads morts sauf 0 réponse + toujours pertinent)
   - Pas un thread concurrent direct en crise (pas de détournement)
3. **Draft** : aide d'abord (étapes concrètes, commandes), mention du
   produit ensuite (1 phrase max, comme contexte pas comme pub).
4. **Validation humaine** : jamais de post auto. L'humain lit et dit « poste ».
5. **Post** : `gh issue comment` (traçable). **Suivi 24h** : répondre à
   toute réaction. **Track R99** sous 5 min.

## Interdits

- Promo sans aide réelle ; lien froid sans contexte technique.
- Plus d'1 intervention par thread (sauf question directe).
- DM aux auteurs/mainteneurs ; brigading (demander des upvotes/stars).
- Réponses générées en masse (1-2 threads/semaine max par projet).

## Référence : Helium (2026-09-28)

Première application : exo#2271 (debug multicast LAN, 0 réponse) et
exo#2299 (shared storage, 0 réponse). Aide technique réelle + mention
contexte. Tracké R99, suivi 24h.
