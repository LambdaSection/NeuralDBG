---
name: rule-113-github-discovery
description: 'RULE 113: GitHub Discovery Protocol — Mesure & Métadonnées'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_113_github_discovery.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 113: GitHub Discovery Protocol — Mesure & Métadonnées

## Rule

La découverte sur GitHub vient de 4 circuits seulement : recherche interne (topics/description), pages topics/Explore (activité), trending (hors de portée d'un petit profil), preuve sociale externe. À ce stade, GitHub **convertit** le trafic, il ne le crée pas. Toute session touchant un repo public DOIT vérifier ses métadonnées de découverte, et toute action marketing (R94/R96) DOIT être mesurée via l'API traffic.

## Verification

### A. Métadonnées (au moins 1×/trimestre par repo public)
```
CHECK: description non vide, topics >= 3, homepage si docs/demo
  gh api repos/{owner}/{repo}
REMEDIATE:
  gh repo edit --description "..." --add-topic ... --add-topic ...
CONTRAINTES: descriptions EN, ASCII uniquement (R39), dérivées du README existant — jamais inventées (R87)
```

### B. Mesure traffic (fenêtre glissante 14 jours — non sondée = perdue)
```
POLL (min 1×/semaine, idéal quotidien):
  GET /repos/{owner}/{repo}/traffic/views
  GET /repos/{owner}/{repo}/traffic/clones
  GET /repos/{owner}/{repo}/traffic/popular/referrers
STOCK: reports/discovery_snapshot.json
NOTE: pas d'API pour les vues de PROFIL (komarev badge = seul proxy)
```

### C. Boucle acquisition (R99)
```
POST (R94/R96) -> +48h: relever referrers -> log correlation dans acquisition_tracker.md
RELEASE/TAG -> créer la GitHub Release (le tag seul ne notifie personne) + draft post R94
```

## Interdits
- Commits fantômes / art sur la contribution graph (TOS GitHub, crédibilité nulle)
- Auto-follow massif / star farming (TOS)
- Posting automatique sur Reddit/Discord (R96 : human-in-the-loop)

## État des lieux 2026-09-17 (preuves)
- 9 repos édités (descriptions EN + topics) : TokenWise, Astral, Datalint, NeuralPaper, Sugar, Odin, kuro-rules, LifeTrack, Lemniscate-world
- Release v1.3.2 NeuralDBG créée (le tag existait orphelin)
- Blog profil : dead-end documenté une fois (token sans scope user) — action manuelle Settings → Profile
- Baseline traffic : NeuralDBG 22 vues / 337 clones (14j), referrers Google + reddit.com

## References
- R94, R96, R99 (flywheel contenu + mesure), R103 (profil), R107 (PR upstream), R39 (encodage), R87 (ownership/preuves)

---
**Created**: 2026-09-17
**Applies to**: All public repos (LambdaSection + Lemniscate-world)
**Enforcement**: MANDATORY
