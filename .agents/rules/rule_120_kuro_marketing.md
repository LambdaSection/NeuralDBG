---
name: rule-120-kuro-marketing
description: Marketing pipeline, human validation gate
trigger: manual
scope: team
priority: 1
status: active
paths:
- docs/**
---

<!-- Generated from rules/rule_120_kuro_marketing.md — DO NOT EDIT. scope=team priority=1 status=active -->

# RULE 120: Marketing via Kuro — Tout Passe par la Pipeline

**Principe** : aucune action marketing (post, réponse, lancement, mesure)
ne se fait hors pipeline Kuro. Les comptes et les clics finaux restent
humains ; tout le reste (drafts, décision, file, tracking) est automatisé.

## Canaux et automatisation

| Canal | Auto | Manuel (humain) |
|---|---|---|
| X hub + annexes | drafts (`gen_x_posts.py`), décision (`post_policy.py`), file + post Buffer gratuit (`buffer_post.py`, `X_VIA=buffer`) | création compte + clés API (téléphone requis), validation du texte |
| Discord | version longue auto (`x_long_*`), post webhook (`kuro_discord.py post`) | création webhook salon (1 min) |
| Reddit | drafts + templates (R96), veille threads (agent-reach) | post + réponses 24h (karma, authenticité, anti-spam) |
| Show HN | kit J-3/J-1 (R95), checklist `R98` | post timing + 2h de réponses (one-shot) |
| Blog | draft (`generate_blog.py`, R110) | relecture + publication |
| Mesure | rappel auto R99 | log `docs/tracking/acquisition_tracker.md` sous 5 min |

## Gates (non négociables)

1. `--dry-run` par défaut partout ; `--apply` n'écrit que des drafts.
2. Un post ne part qu'avec texte validé humainement (jamais de draft brut moteur).
3. Anti-leak R94-v2 (sanitize + worthiness + jamais de hash/préfixe brut).
4. `X_PLAN=free` par défaut : zéro appel API X payant sans décision explicite.
5. Dedupe par hash (`x_posted.json`) : jamais deux fois le même post.

## Nouveau projet : branchement en 3 étapes

1. Compte X dédié + 4 clés dans `.env` (`X_<SLUG>_*`), ou canal Buffer (`BUFFER_CHANNEL_<SLUG>`).
2. Webhook Discord du salon (optionnel mais gratuit).
3. `gen_x_posts.py --apply --only <Projet>` + `post_policy.py --dry-run --annex <Projet>` pour vérifier.

## Référence : Helium (2026-09-22)

Premier projet branché de bout en bout : draft quotidien auto, premier post
hub via Buffer (id `6ab2dfc896bce92ace641d24`), rewrite humain obligatoire
constaté (voix changelog rejetée), tracking R99 à jour.
