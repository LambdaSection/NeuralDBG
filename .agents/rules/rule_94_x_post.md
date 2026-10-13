---
name: rule-94-x-post
description: 'RULE 94: Daily X Post — Obligation de Publication Quotidienne'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_94_x_post.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 94: Daily X Post — Obligation de Publication Quotidienne

Trigger: à la fin de CHAQUE session de travail (obligatoire, comme R83).

---

## Purpose

Maintenir une présence publique quotidienne sur X (anciennement Twitter).
Chaque session de travail doit produire 1 post publiable, peu importe la taille
de la session.

---

## Format

Un post unique, max 280 caractères, formaté ainsi :

```
[emoji] NeuralDBG: [résultat clé de la session]

[une métrique ou un chiffre fort]

[#NeuralDBG #ML #DeepLearning]
```

---

## Règles de contenu

1. **Ne JAMAIS** révéler le code propriétaire, les heuristiques causales, ou les
   détails implémentatoires de NeuralDBG-Engine
2. **Toujours** inclure une métrique chiffrée (ex: "135 tests pass", "0.905 accuracy",
   "5 architectures dogfoodées", etc.)
3. **Toujours** mentionner un livrable concret (demo, benchmark, test, etc.)
4. **Jamais** de promesses sur des dates ou features futures
5. **Ton** : technique, factuel, pas de hype
6. **Hashtags** : minimum 2, maximum 3 (#NeuralDBG + domaine concerné)

---

## Exemple

```
NeuralDBG: LoRA fine-tuning dogfooding done — 3 failure scenarios
(NaN, exploding, catastrophic forgetting) now tracked.

138 tests pass, 6 architectures couvertes.

#NeuralDBG #LLM #FineTuning
```

---

## Emplacement

- Le post est sauvegardé dans `outputs/x_post_YYYY-MM-DD.md`
- Le fichier contient uniquement le texte du post (pas de métadonnées)
- Prêt à être copié-collé sur X

---

## Sanction

Si une session se termine sans post X (ou sans raison valide documentée),
l'agent DOIT le signaler comme tâche non complétée dans le résumé de session.

---

# RULE 94-v2: X Multi-Comptes Top-Vélocité — Kuro (2026-09-21)

## Principe

R94 historique (1 post hub/session dans `outputs/x_post_YYYY-MM-DD.md`) reste valide.
R94-v2 ajoute la couche multi-projets **sans exploser à 60 comptes** :

- **Hub unique** : 1 post/jour agrégateur (top vélocité du jour).
- **Comptes dédiés** : uniquement top-3 vélocité du jour, statut Actif/Validation,
  ownership OWNED (R87), dernier commit ≤ 30j. Les autres jours : skip, pas de post.
- **EXTERNAL / UNKNOWN / Archive / En Pause / Outil** : jamais de compte dédié.
  Mention hub uniquement.

## Sélection automatique (déterministe, sans LLM)

Source unique : parser `generate_portfolio.parse_epingle` + `git log` :

```
score = min(30, commits_30j * 3) + récence (7j +10, 30j +7, 60j +3, 90j -5, >90j -15)
éligible = statut in (Actif, Validation) AND ownership == OWNED AND NOT section-externe
tri décroissant score → top-N (défaut 3)
```

Ownership (miroir `KuroUtils.psm1` + orgs satellites λ, quantifiés `git remote -v` le 2026-09-21) :
- OWNED : `Lemniscate-world/`, `Lemniscate-SHA-256/`, `pbakaus/`, `LambdaSection/`,
  `Quant-Search/` (λ-2), `AI8-Algorithm-Intelligence-Section-8/` (λ-8),
  `Hackin-Life-X/` (λ-3), `N-Hypatia/` (λ-12), `EpureCAD/` (λ-15),
  `HeliumXChain/` (λ-7)
- EXTERNAL : `Demeter-Financial-Labs/` → exclu
- UNKNOWN (pas de remote) → exclu, clarification humaine requise

## Radar IA : sélectif par construction

`scripts/kuro_radar.py` ne touche jamais tous les repos : Archive exclu,
1 projet max par signal, max 8 recos règles + 8 IA. R94-v2 réutilise ce mapping :
X ne poste que si vélocité + match radar convergent. Jamais de blast global.

## Pipeline Kuro (anti-redondance)

Fait partie du **robot unique** `.github/workflows/kuro.yml` (vision Kuro § anti-redondance).
Interdiction de créer un 2e workflow X. Étape après `Blog factuel` :

```
audit_truth_daily → compute_progress → gen_x_posts.py --top 3 --apply
→ outputs/x_post_YYYY-MM-DD-{projet}.md + outputs/assets/ + hub x_post_YYYY-MM-DD.md
→ commit unique kuro-rules (single writer)
```

Local : Daemon + Desk pour preview, API `/api/projects/{name}/velocity` comme source.
Script : `scripts/gen_x_posts.py` (stdlib uniquement, cross-platform R93, logs ASCII).

## Full-auto principal vs annexes — politique (2026-09-21, gate 14j levé par l'utilisateur)

Décision sans humain via `scripts/post_policy.py` (état `outputs/x_posted.json`) :

| Compte | Quand | Garde-fous |
|---|---|---|
| PRINCIPAL (hub) | 1/jour max = top vélocité, hash jamais posté | mêmes gates R94-v2 |
| ANNEXE (ex: Helium) | score ≥ 15 ET cooldown 48h ET hash jamais posté sur ce compte | `--annex Helium` déclare le compte |

```
python scripts/post_policy.py --dry-run --top 5 --annex Helium
python scripts/post_policy.py --apply --top 5 --annex Helium   # poste (clés requises)
```

Clés par compte : hub = `X_*`, annexe = `X_<SLUG>_*` (ex: `X_HELIUM_API_KEY`,
voir `.env.example`). Sans clés : draft seul, jamais de post. Chaque post est
loggé dans `outputs/x_posts_log.md` + `x_posted.json` (dedupe) ; report R99
dans `acquisition_tracker.md` lors de la revue hebdo.

## Plan X free (2026-09-21 : pay-per-use depuis 02/2026, plus de free tier)

Poster via l'API coûte ~0,015 $/post sans lien (~0,45 $/mois à 1/jour), 0,20 $
avec lien (nos drafts n'ont jamais de lien). `X_PLAN=free` (défaut) = drafts
locaux + posts Discord auto, zéro appel X (pas d'erreur 402).
`X_PLAN=basic` = posting actif. Vérifié live le 2026-09-21 (402 sans crédits).

## Buffer gratuit — posting X sans payer (2026-09-21)

Plan free : 3 canaux, 10 posts en file par canal. `scripts/buffer_post.py`
(API GraphQL, clé `BUFFER_API_KEY`) + `post_policy --via buffer`
(canaux `BUFFER_CHANNEL_HUB`, `BUFFER_CHANNEL_HELIUM`). File pleine =
skip propre avec raison (jamais de crash). `X_VIA=buffer` par défaut
dans la tâche quotidienne.

## Format (2026-09-21 : explicatif, plus du commit brut)

Hub LambdaSection en anglais (exemple réel), annexes en français formel :
verbes d'action (Shipped/Fixed/Sped up), statuts traduits, présentations EN,
détection du franglais pénalisée au score (thèmes FR traduits par le rewrite
LLM quand le moteur répond, sinon signalés dans l'historique).

```
Helium (blockchain Rust) — Nouveau : Helium-mesh MVP + one-click install + releases.

3 commits 30j, 1 à 7j, 28% Actif.

#Helium #Rust #Blockchain
```

Ligne 1 = Projet (présentation une-ligne, français formel accessible non-expert —
réf : `MARKETING_MEMORY/helium-post-format-formel.md`) + voix du jour (log,
leçon, chiffre, question en rotation déterministe) : sujet reformulé en phrase
naturelle. Version longue Discord = présentation + travaux numérotés +
prochaine étape (SESSION_SUMMARY, sinon repli honnête) + caption.
**Jamais de hash, jamais de préfixe brut.** Métrique + statut
ligne 2, lien repo vérifié ligne 3, 2-3 hashtags ligne 4. ≤ 280 caractères
(comptage t.co : URL = 23), zéro promesse future.
Lisibilité Flesch (1948) dans le score du moteur ; réécriture LLM auto
(`kuro_llm`) gardée UNIQUEMENT si lint OK + score ≥ + chiffres/liens préservés.

## Worthiness — on ne poste que ce qui informe (2026-09-21)

Un rename interne ou un chore ne part jamais : `week_themes()` agrège les
commits porteurs de sens des 7j (`feat/perf/refactor`, `fix` non-trivial),
filtre le trivia (pre-commit, merges, typos, renommages) et refuse le vide
(`rien-de-publiable`, ni X ni Discord). Coupe toujours au mot près.

## Liens vers nos projets (2026-09-21)

Chaque post porte le lien du repo **vérifié vivant** (`gh api`, `PROJECT_LINKS` —
jamais deviné, jamais de 404) + version longue Discord avec `Repo : <url>`.
Les URLs comptent 23 car (t.co) dans la limite 280 (`x_len()`).
Pas de lien blog : les `.md` ne sont pas servis en pages (404 vérifié).
Coût : 0,20 $/post avec lien en API directe, **0 $ via Buffer**.

## Lien en reply — anti-déclassement algo (2026-09-21)

X déclassifie les posts avec lien sortant. Par défaut : thread auto
(post nu + reply `Code et détails ici : <url>`, `buffer_post.create_thread`).
Le lien reste dans les drafts (fallback si thread impossible) et dans les
versions Discord. Comptage t.co conservé sur chaque partie (≤280).

## Moteur posts — raisonne et s'améliore (`scripts/post_brain.py`, 2026-09-21)

- `review --project X --from-file <draft>` : score /100 (checks + critique LLM
  OpenRouter→DeepSeek→Ollama, best-effort, ne bloque jamais) + `--rewrite`
  gated (lint + score). Chaque review est mémorisée (`post_history.jsonl`).
- `flag --project X --term <mot>` : muselle un terme en 1 commande
  (`config/posting_overrides.json`, additif, effectif dès demain).
- `ingest` : statut réel file→envoyé→disparu via Buffer (`post_delivery.json`).
- `learn` : constats chiffrés + propositions depuis l'historique.
- Hook consultatif dans `post_policy` (score loggé, jamais bloquant).

## Stratégies de post par projet — anti-leak (2026-09-21)

Chaque projet a une stratégie (`POSTING_STRATEGIES` dans `gen_x_posts.py`) :
`reveal` (chiffres processus autorisés) + `never` (termes bloquants) + fallback générique.

- **Couche 1 — universelle, TOUS projets** : emails, secrets (`token=`...), chemins
  locaux, blobs hexadécimaux ≥ 32 car, téléphones, montants €/$, IP privées.
- **Couche 2 — par projet** : OpenQuant (40+ termes : alpha, stats, corrections
  multi-tests, bootstrap, contrefactuel), NeuralDBG-Engine
  (heuristiques, R94 §1), Forma (données assurés), LifeTrack (données santé),
  Helium (seeds/clés), Horcruxe Labs (pistes de publication).

- **OpenQuant (trading)** : processus OK (commits, tests, gates, coverage),
  JAMAIS d'alpha (edge, signaux, sharpe, drawdown, sizing, leverage, backtests...).
  `lint_post()` bloque tout draft contenant un terme ; `format_post_safe()`
  génère alors un draft générique (type de changement + chiffres, zéro message brut).
  `post_policy --apply` re-vérifie et refuse (`LINT-BLOCK`).
- **Défaut** : sanitization chemins/clés + ton factuel, fallback générique actif.
- **Discord** : version longue auto (`x_long_YYYY-MM-DD-{projet}.md` :
  présentation + travaux + caption, jamais de promesse future) pour les salons.

## Gate de publication (sécurité)

- `--dry-run` par défaut. `--apply` écrit les drafts, **ne poste rien**.
- Full-auto X interdit tant que 14j consécutifs sans leak (R94 §1 : jamais de code
  proprio, clés, tokens, paths). `sanitize()` redacte `api_key/token/secret/password`
  et les chemins locaux.
- Chaque publication manuelle postée DOIT être loggée dans
  `docs/tracking/acquisition_tracker.md` sous 5 min (R99).
- Format draft : ≤ 280 caractères, 1 métrique chiffrée, 1 livrable concret
  (hash + message commit), 2-3 hashtags, ton technique factuel, zéro promesse future.
- Voix : `MARKETING_MEMORY/post-format-formel-global.md` + historique `outputs/x_post_*.md`.

## Compte par projet (ex : Helium) — procédure

Créer un compte X est manuel (vérification téléphone par compte, impossible à automatiser) :

1. Créer le compte (ex : Helium), puis sur https://developer.x.com : projet + app,
   User authentication → Read+Write, générer API Key/Secret + Access Token/Secret
   **du compte du projet** (1 compte = 1 jeu de clés).
2. Mettre les 4 clés dans `.env` (`X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`,
   `X_ACCESS_SECRET` — voir `.env.example`). Jamais commité (R10/R76).
3. Draft ciblé (mêmes gates, sans cutoff top-N) :
   `python scripts/gen_x_posts.py --apply --only Helium`
4. Vérifier puis poster :
   `python scripts/x_post.py --from-file outputs/x_post_YYYY-MM-DD-helium.md --dry-run`
   `python scripts/x_post.py --from-file outputs/x_post_YYYY-MM-DD-helium.md --apply`
5. Logger dans `docs/tracking/acquisition_tracker.md` sous 5 min (R99).

## Discord : même logique (Kuro suit Epingle)

- `python scripts/kuro_discord.py plan` — catégories + salons proposés (21 salons).
- `python scripts/kuro_discord.py sync --guild ID --apply` — crée et corrige
  (nécessite `DISCORD_BOT_TOKEN`, voir `.env.example` étapes 1-5).
- `python scripts/kuro_discord.py post --project Helium --message-file <draft>` —
  route vers le webhook du salon (config `kuro_discord_channels.local.json`).
