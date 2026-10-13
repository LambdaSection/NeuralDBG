---
name: rule-121-pr-fix-marketing
description: 'PR-fix marketing: real fix + traceable PR'
trigger: manual
scope: team
priority: 1
status: active
paths:
- docs/**
---

<!-- Generated from rules/rule_121_pr_fix_marketing.md — DO NOT EDIT. scope=team priority=1 status=active -->

# RULE 121: PR-Fix Marketing — Reparer en Public avec nos Outils (MANDATORY)

**Principe** : la preuve bat le discours. Chaque semaine, on repare un vrai probleme ouvert d'un repo cible avec nos outils, en public, avec le produit en preuve — pas en slogan. Complement de R119 (aide par commentaire) et R107 (PR upstream sur bugs de notre catalogue) : ici on vise les issues et PRs ouvertes des autres, fix reel + PR ou review traçable.

## Position vs R107 / R119

| | R119 (commentaire) | R121 (cette regle) | R107 (upstream) |
|---|---|---|---|
| Quoi | Reponse d'aide sur issue ouverte | Fix reel ou review de PR sur repo tiers | PR sur bug de notre catalogue |
| Livrable | Commentaire `gh issue comment` | PR soumise ou review postee | PR mergée upstream |
| Metrique | Reponses recues, conversions pilote | PRs soumises, mergees, reactions | Merge rate > 50% |
| Cadence | 1-2 threads/semaine/projet | 1-2 PRs ou reviews/semaine/projet | Sous 48h quand bug confirme |

Les trois tournent en parallele, jamais l'un sans suivi 24h.

## Processus hebdomadaire (obligatoire)

1. **Veille** (automatise) : `gh search issues` + `gh search prs` sur les mots du domaine (ex : NaN loss, vanishing gradient, backtest overfitting, walk-forward, shared storage, discovery LAN). Priorite aux repos avec audience (stars, watchers) dans notre niche.
2. **Qualification** (tout DOIT etre vrai) :
   - Issue ouverte avec 0 a 2 reponses, ou PR ouverte bloque sur un bug reproduisible
   - Probleme qu'on a VRAIMENT resolu avec notre outil (preuve : logs, sortie NeuralDBG/OpenQuant, tests verts)
   - Moins de 6 mois d'inactivite (exception : 0 reponse + toujours pertinent)
   - Pas un thread concurrent direct en crise (pas de detournement), pas de prise de controle hostile de PR
   - Le fix tient en diff petite et reversible (pas de refactor impose)
3. **Repro locale** : reproduire le bug hors repo cible (script ou notebook), montrer la detection par notre outil (ex : NeuralDBG : module responsable + confiance ; OpenQuant : DSR/PBO avant-apres). Jamais de donnee privee, jamais de secret (R41, R76).
4. **Draft** : aide d'abord (etapes concretes, commandes, test), mention du produit ensuite (1 phrase max, comme contexte pas comme pub). Template description PR : probleme, repro, cause racine, fix, preuve outil, tests.
5. **Validation humaine** : jamais de PR ni de review auto. L'humain lit le diff et dit "poste". Gate R120 non negociable.
6. **Post** : `gh pr create` ou `gh pr review` / `gh issue comment` (traçable). **Suivi 24h** : repondre a toute reaction. Si ignore 7 jours : 1 ping poli max (cf R107). **Track R99** sous 5 min dans `docs/tracking/acquisition_tracker.md`.
7. **Capitalisation** : si merge ou discussion positive, ajouter le lien au README/portfolio (R103) et au plan leads (offre beta). Si refuse, noter la raison (lecon, pas d'insistance).

## Interdits

- PRs de masse, nitpicks cosmétiques ou formatting seul pour faire du volume.
- Plus d'1 PR ou review par thread (sauf demande directe du mainteneur).
- DM aux auteurs/mainteneurs ; brigading (demander upvotes/stars).
- Promo sans fix reel ; lien froid sans repro technique.
- Auto-post sans validation humaine (R120 gate 2). Les clics finaux restent humains ; drafts et veille sont automatises.
- Cross-post identique sur plusieurs repos.

## Verification

```
Chaque semaine :
  IF 0 PR/review R121 postee ET 0 thread R119 traite :
    VIOLATION: marketing technique a l'arret
    ACTION: 1 cible qualifiee avant la fin de semaine + log R99

Apres chaque PR/review :
  VERIFY: commentaire ou PR traçable par URL
  VERIFY: entree R99 sous 5 min (plateforme, lien, resultat)
  VERIFY: diff relu humainement avant envoi (jamais de draft brut moteur)
```

## Reference : Helium (2026-09-28)

Premiere application R119 (non R121) : exo#2271 et exo#2299, aide technique + mention contexte, tracke R99, suivi 24h. R121 demarre au prochain fix avec diff : objectif 1 PR soumise/semaine/produit actif (NeuralDBG, OpenQuant).

---

**Created**: 2026-10-02
**Trigger**: demande user — transformer les fixes publics en strategie marketing systematique
**Applies to**: tous produits actifs avec preuve outil (NeuralDBG, OpenQuant, Helium)
**Enforcement**: MANDATORY
**Pairs with**: R119 (commentaires), R107 (upstream PRs), R120 (pipeline), R99 (tracking), R111 (donnees locales)
