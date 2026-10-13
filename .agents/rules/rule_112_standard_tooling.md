---
name: rule-112-standard-tooling
description: 'RULE 112: Standard Tooling — Agent-Reach + Codebase-Memory sur chaque
  projet — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_112_standard_tooling.md — DO NOT EDIT. scope=company priority=2 status=active -->

## RULE 112: Standard Tooling — Agent-Reach + Codebase-Memory sur chaque projet — MANDATORY

### Rule

Toute nouvelle création de projet (repo, service, outil) doit intégrer les deux
capacités standard de l'entreprise lambda-Section dès sa création :

**1. Mémoire structurelle — codebase-memory-mcp (local, privé)**

- L'indexation est automatique à la première connexion d'agent si
  `auto_index = true` (déjà actif sur la machine principale).
- Si l'index n'existe pas : `codebase-memory-mcp cli index_repository
  '{"repo_path": "<chemin>", "project": "<nom>"}'`.
- Explorer le code via le graphe (search_graph, trace_path, get_architecture)
  AVANT de tomber dans grep/read consommateur de tokens.
- Recommandé : committer l'artefact d'équipe `.codebase-memory/graph.db.zst`
  (les co-équipiers sautent le réindex) ; ajouter `.codebase-memory/` au
  `.gitignore` si l'on préfère que chacun indexe localement.

**2. Accès internet — Agent-Reach (recherche et veille)**

- Toute phase de recherche/veille (R69 Intelligence Harvester, R75 Desk
  Research) passe en priorité par les canaux Agent-Reach : Jina Reader (web),
  Exa (recherche sémantique), RSS, YouTube, GitHub, V2EX, Bilibili.
- Vérifier la santé : `agent-reach doctor`.
- Les canaux à login (Twitter, Reddit, 小红书…) ne s'activent qu'avec
  l'accord explicite de l'utilisateur et un compte dédié (jamais un compte
  principal).

### Verification

```
ACTION: à la création d'un projet, vérifier :
VERIFY: 1. `codebase-memory-mcp cli index_status` montre le projet indexé
        2. `agent-reach doctor` au moins 4 canaux verts
        3. exploration code via graphe avant grep massif
```

### Enforcement

IF un projet est créé sans index CBM :

- indexer immédiatement (commande ci-dessus), ne pas commencer l'exploration sans

IF une recherche web est faite par scraping manuel alors qu'Agent-Reach est disponible :

- STOP, refaire via les canaux Agent-Reach

IF un canal à login est demandé :

- demander le choix de l'utilisateur + rappeler le conseil du compte dédié

---
