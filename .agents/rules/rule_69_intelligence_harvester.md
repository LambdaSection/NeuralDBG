---
name: rule-69-intelligence-harvester
description: 'RULE 69: Intelligence Harvester — Collecte de Sources Externes'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_69_intelligence_harvester.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 69: Intelligence Harvester — Collecte de Sources Externes

## Trigger
À chaque milestone de projet (L1 → L2, L2 → L3, avant tout lancement public) + **avant chaque phase build (S1/S2/S3) + hebdo si Technical Risk = HIGH**.

## Trigger additionnel: Continuous Tech Scout (MANDATORY si R14.5 Technical = HIGH/MEDIUM)

Avant chaque phase build et chaque semaine d'execution, l'agent DOIT harvester 3+ repos/techno recents qui reduisent le risque technique. Sources prioritaires: GitHub Trending (rust, wireguard, firecracker, libp2p), papers-with-code, HN, crates.io.

## Objectif
Collecter 3+ sources externes (citations, études, posts, données) qui prouvent ou infirment une hypothèse clé du projet.

## Sources autorisées (par priorité)

| Priorité | Source | Type de preuve |
|---|---|---|
| 1 | Papers académiques (arXiv, ACL, NeurIPS) | Preuve forte, peer-reviewed |
| 2 | Données officielles (entreprises, régulateurs) | Preuve factuelle |
| 3 | Posts techniques (Reddit, HN, X, blogs) | Signal qualitatif |
| 4 | Études de marché (Gartner, Statista, rapports) | Preuve quantitative |
| 5 | Interviews / expert calls | Preuve primaire |
| 6 | Repos GitHub / crates.io / papers-with-code (Tech Scout) | Preuve d'implementation, reduit Technical Risk |

## Tech Scout — Processus (obligatoire)

1. Identifier le blocage technique (ex: "NAT traversal WireGuard", "Firecracker rootfs build")
2. Chercher 3+ repos recents (<12 mois, >100 stars ou last commit <30j) qui resolvent le blocage
3. Pour chaque repo, noter:
   - URL / stars / last commit / license
   - Ce qu'il resout (ex: wireguard-go userspace = pas besoin kernel)
   - Limites (ex: pas de KVM sur Windows)
   - Snippet d'integration (1-3 lignes)
4. Ajouter dans `research/tech_scout.md` (creer si inexistant) + lier dans `desk_research_report.md` Risk Analysis Remedy
5. Si 3+ repos trouves avec last commit <30j et >500 stars: Technical Risk peut etre downgrade HIGH->MEDIUM (documenter)

## Autorisation permanente "en apprendre" (2026-10-04)

Quand l'utilisateur dit "en apprendre <url github>" ou "on peut apprendre quoi de ceci ? <url>",
l'agent est autorise en permanence a cloner le repo en lecture seule pour apprentissage,
sans redemander :

1. `git clone --depth 1 --filter=blob:none --sparse <url>` dans `%TEMP%/opencode` ou `/tmp`
   (jamais dans le workspace sauf demande explicite), `sparse-checkout` minimal
   (`docs`, `crates/.../embedding_host`, `README.md`) pour rester leger.
2. Lecture seule : jamais de commit/push/fork, jamais d'exfiltration de secrets,
   jamais d'execution de code non revu du repo clone.
3. Restituer : URL / stars / license / ce qu'il resout / limites / snippet 1-3 lignes,
   comme un Tech Scout.

## Processus

1. Identifier l'hypothèse à vérifier (ex: "les devs ML galèrent avec le debugging de gradients")
2. Chercher 3+ sources qui y répondent
3. Pour chaque source, noter :
   - URL / citation exacte
   - Date de publication
   - Ce qu'elle prouve
   - Ce qu'elle ne prouve PAS
4. Ajouter les résultats dans le fichier evidence approprié (evidence-matrix.csv, desk_research_report.md)

## Emplacement des données

Les sources collectées sont stockées dans :
- `kuro-rules/KNOWLEDGE_BASE/mom_tests/` (si issues d'interviews)
- `kuro-rules/MARKETING_MEMORY/` (si issues de desk research)
- Le fichier `research/evidence-matrix.csv` du projet concerné
- `research/tech_scout.md` (Tech Scout: 3+ repos recents, obligatoire avant chaque phase build si Technical HIGH)

## Integration avec R14.5 et R75

- Avant chaque phase build: Tech Scout doit alimenter la colonne `Evidence AGAINST` et `Remedy` du 5-risk table (R14.5)
- Dans R75 Risk Analysis: citer 1+ repo Tech Scout comme remedy
- Si aucun repo recent trouve apres 2h recherche: documenter comme risque confirme (Technical HIGH maintenu)

## Sanction

Si un milestone est passé sans 3+ sources collectées :
- Le projet est rétrogradé au niveau de validation précédent
- L'agent DOIT documenter pourquoi les sources manquent
- Aucun lancement public autorisé tant que les sources ne sont pas collectées

## Fichiers associés
- `research/evidence-matrix.csv` — matrice des preuves
- `research/scorecard.md` — scorecard du projet
- `docs/desk_research_report.md` — rapport de desk research
