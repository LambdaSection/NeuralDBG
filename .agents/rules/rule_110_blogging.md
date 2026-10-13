---
name: rule-110-blogging
description: 'RULE 110: Blogging & Contenu Public — le système d''écriture lambda-Section'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_110_blogging.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 110: Blogging & Contenu Public — le système d'écriture lambda-Section

**Statut : MANDATORY pour tout article, billet de blog, thread long ou tutoriel publié au nom de lambda-Section.**
Complète : R75 (desk research → sources), R83 (résumés investisseurs), R94 (X posts quotidiens),
R96 (posts communautaires), R99 (acquisition tracker). Le pipeline `scripts/generate_blog.py`
produit des brouillons factuels ; la narration humaine (ou assistée) suit cette règle.

## R110.1 — Vérité d'abord

- Tout chiffre cité provient du pipeline vérité (git log, tests, Epingle) ou d'une
  source citée par lien. Aucune promesse, aucun % inventé.
- Les données financières, clients et vulnérabilités exploitables ne sont JAMAIS
  publiées (voir R110.5).

## R110.2 — Types de posts (choisir avant d'écrire)

| Type | Usage | Longueur |
|---|---|---|
| **Deep-dive** | "Comment ça marche vraiment" (modèle Julia Evans / Dan Luu) | 1500+ mots |
| **Build log** | Avancement factuel d'un projet, décisions + trade-offs | 600–1000 |
| **Tutorial** | Étapes reproductibles, code exécutable complet | 800–2000 |
| **Opinion** | Prise de position argumentée sur l'industrie | 500–1000 |
| **Post-mortem** | Ce qui a cassé, pourquoi, ce qui change | 600–1200 |

Cadence : la régularité prime sur la fréquence (mensuel constant > rafale puis silence).

## R110.3 — Structure obligatoire

1. **Ouverture = problème** : scénario vécu, conséquence du problème, ou affirmation
   surprenante. Interdits : définition de dictionnaire, "Dans cet article nous allons…",
   résumé du sommaire.
2. **TL;DR en tête** (2-3 phrases) pour les posts > 800 mots.
3. **Titres H2 riches et spécifiques** : "Pooling de connexions avec pgBouncer" et non "Étape 3".
4. **Code exécutable copiable** toutes les 3-5 paragraphes ; imports et contexte inclus ;
   sinon lien vers un repo complet. Le pseudocode est interdit.
5. **Tableaux pour toute comparaison.**
6. **Ordre pyramide inversée** : solution/insight d'abord, contexte ensuite.
7. **Gotchas & limites** avant la conclusion (c'est là que l'expérience se voit).
8. Conclusion qui synthétise, ne répète pas.

## R110.4 — Style

- Voix directe, 2ᵉ personne ("vous"), conversationnelle mais précise.
- Zéro gatekeeping ("évidemment", "il suffit de" interdits).
- Chaque claim sourcé : lien officiel, étude, benchmark, ou nos propres mesures.
- Édition impitoyable : une passe "suppression", une passe "précision", lecture à voix haute.
- Titre spécifique et recherchable ("Ce que la migration X→Y nous a appris" > "Retour d'expérience").

## R110.5 — Limites du build in public

NE JAMAIS publier :
- Données clients (même anonymisées) sans consentement explicite
- Détails de sécurité exploitables (stack auth, tokens, infra interne)
- Roadmap détaillée avec timelines (donne un playbook aux concurrents)
- Posts émotionnels / drama (clients, concurrents)
- Données financières brutes au-delà des métriques volontairement publiques

Partager en priorité : décisions + trade-offs, leçons techniques, milestones,
métriques choisies, coulisses du process.

## R110.6 — SEO & distribution

1. Version canonique auto-hébergée (Lemniscate-world/blog), métadescription + OG image.
2. Syndication dev.to/Hashnode/Medium avec `rel=canonical` pointant vers l'original.
3. X : thread de ~5 tweets extrayant les findings, lien en dernier tweet.
4. Hacker News : mardi-jeudi matin US, contenu technique profond uniquement.
5. Reddit : lire la culture du sub une semaine avant de poster ; texte > lien selon les règles.
6. LinkedIn : angle architecture/décisions pour audience senior.
7. Discord lambda-Section d'abord (communauté propre), puis communautés tierces où l'on
   est membre actif.

## R110.7 — Checklist de publication (12 points)

- [ ] Titre spécifique et cherchable
- [ ] Premier paragraphe clair sur le pourquoi
- [ ] Sommaire si post long
- [ ] Pourquoi → Quoi → Comment
- [ ] Diagramme pour chaque concept complexe
- [ ] Code réellement exécutable
- [ ] Alternatives discutées (pourquoi pas elles)
- [ ] Chiffres mesurés, pas estimés
- [ ] Takeaways explicites pour le lecteur
- [ ] Sources originales liées
- [ ] Orthographe + liens vérifiés
- [ ] Meta description + OG image

## R110.8 — Skills externes officiels (à consulter avant d'écrire)

- Developer Writing Guide (patterns, scanning, distribution) :
  https://unmarkdown.com/blog/developer-writing-guide
- Complete Developer Writing Guide (design docs → blogs → talks) :
  https://www.youngju.dev/blog/culture/2026-04-15-developer-writing-complete-guide-design-doc-rfc-blog-book-conference-talk-deep-dive-guide-2025.en
- How to Write a Technical Blog Post That People Actually Read :
  https://firstdev.blog.rsynk.com/tips/writing-technical-blog-posts-developers
- Draft.dev Technical Blogging Style Guide :
  https://draft.dev/learn/styleguide
- Technical Writing for Developers (pyramide inversée) :
  https://novvista.com/the-developers-guide-to-technical-writing-from-code-comments-to-blog-posts/
- Build in Public Guide (partager / ne pas partager) :
  https://www.makrly.com/blog/build-in-public-guide

Condensé local : `KNOWLEDGE_BASE/BLOGGING/CORE.md`.

## R110.9 — Violations

- Chiffre non sourcé par le pipeline vérité → VIOLATION.
- Ouverture "définition/sommaire" → VIOLATION.
- Publication sans checklist R110.7 → VIOLATION.
- Fuite d'une catégorie R110.5 → VIOLATION CRITIQUE.

---
**Créée** : 2026-08-22 — issue recherche web (sources R110.8)
**Applies to** : tout contenu public lambda-Section
**Enforcement** : MANDATORY
