---
name: rule-109-adaptive-design
description: 'RULE 109: Adaptive Design Systems — identité par produit, socle universel'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_109_adaptive_design.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 109: Adaptive Design Systems — identité par produit, socle universel

**Statut : MANDATORY pour toute surface UI de tout projet lambda-Section.**
Complète R108 (Ledger Brutal = langage réservé à LifeTrack & dérivés) et R72 (Impeccable = base process).
Principe fondateur : **un produit = une identité choisie, pas un template répliqué.**

## R109.1 — Choisir le langage AVANT de designer

Avant toute UI, l'agent déclare dans un `DESIGN.md` à la racine du produit :
le langage choisi, la raison, les tokens racines. Matrice de sélection :

| Contexte produit | Langage de référence | Signaux |
|---|---|---|
| Devtool / audience dev | GitHub Primer + mono dense | docs, CLI, dashboards techniques |
| Data enterprise dense | IBM Carbon (density, a11y CI) | tables, monitoring, back-office |
| Mobile-first grand public | Apple HIG ou Material 3 | iOS/Android natif, grand public |
| Fintech / confiance | Shopify Polaris (clarté marchand) | paiement, transactions |
| Éditorial / portfolio / marque forte | Ledger Brutal (R108) ou éditorial serif | vitrine, storytelling |
| Produit Apple-centrique | HIG strict (Clarity/Deference/Depth) | écosystème Apple |

Deux produits différents NE partagent PAS leur skin par défaut. La cohérence
inter-produits vient du socle R109.2, pas d'un copier-coller visuel.

## R109.2 — Socle non négociable (tous langages confondus)

1. **Tokens 3 tiers** : primitives (`blue-500`) → sémantiques (`color.action.primary`)
   → composant (rare). Les composants ne consomment QUE le sémantique.
   Format : W3C DTCG (pin une version datée, ex 2025.10). Jamais de hex brut en composant.
2. **Rôles de couleur** : `primary/on-primary`, `surface/on-surface`, `container` —
   jamais `blue` côté usage. Un rebrand = rebind des sémantiques, zéro diff composants.
3. **Contraste WCAG 2.2 AA vérifié en CI PAR PAIRE SÉMANTIQUE** (4.5:1 texte,
   3:1 large/non-texte), pour chaque thème indépendamment. Pas d'œil nu.
4. **Thèmes frères** : light/dark/high-contrast = jeux de valeurs des mêmes clés
   sémantiques. Honorer `prefers-color-scheme`, `prefers-contrast`, `forced-colors`.
5. **Motion tokenisée** : courbes nommées + durées ; variante `prefers-reduced-motion`
   générée automatiquement. Une seule famille d'easing par produit.
6. **Cible tactile ≥44px** sur toute cible interactive mobile.
7. **A11y en CI, pas en fin de projet** (modèle Carbon) : chaque contribution passe
   les checks automatiques (contraste, focus, clavier).
8. **Anti-slop (hérité R108)** : pas de gradient décoratif, glassmorphism, glow,
   violet-par-défaut, emoji décoratif — SAUF si le langage choisi du produit le
   définit explicitement (ex: Liquid Glass sur un produit Apple-style).

## R109.3 — Skills externes officiels (à consulter avant tout travail UI)

- W3C Design Tokens spec : https://designtokens.org (pin version datée)
- Style Dictionary (build tokens multi-plateformes) : https://styledictionary.com
- Material 3 (HCT, rôles tonals, Theme Builder) : https://m3.material.io
- Apple HIG : https://developer.apple.com/design/human-interface-guidelines
- IBM Carbon (a11y + tokens 3 tiers documentés) : https://carbondesignsystem.com
- GitHub Primer (devtools) : https://primer.style
- Shopify Polaris : https://polaris.shopify.com
- Radix primitives (a11y comportementale) : https://radix-ui.com
- shadcn/ui conventions (code possédé, fork > wrapper) : https://ui.shadcn.com
- Playbook migration/refactor tokens (inventory→guardrails→batch PR) :
  https://rockpaperscissors.studio/how-to-refactor-a-design-system-a-practical-7-step-playbook/
- Tokens qui survivent au handoff (naming, contrast-matrix CI) :
  https://www.72technologies.com/blog/design-tokens-that-survive-engineering
- Theming agentic (3 tiers, thèmes frères, forced-colors) :
  https://agenticdevelopercookbook.com/guidelines/implementing/ui/theming-with-tokens

- Build in Public Guide (partager / ne pas partager) — voir R110.8
- Three.js / React-Three-Fiber (3D web immersive) : https://threejs.org · https://r3f.docs.pmnd.rs
  Réservé aux surfaces où la 3D sert le produit (mondes éditoriaux, data-viz spatiale,
  démos hardware). Garde-fous obligatoires : fallback statique sans WebGL,
  `prefers-reduced-motion` respecté (scène figée), budget perf (LCP < 2.5s),
  aucune information portée uniquement par le mouvement.

## R109.4 — Workflow agent obligatoire

1. Lire le `DESIGN.md` du produit (ou le créer selon R109.1) AVANT toute UI.
2. Inventory avant refactor : tokens existants, valeurs brutes, dérives.
3. Migrations par petites batches reviewables (1 famille de tokens par PR),
   jamais de grand-bang autonome.
4. Toute règle doit être testable : lint anti-valeurs-brutes, check contraste
   par paire sémantique, naming convention documentée avec exemples valides/invalides.
5. Critique loop : après génération UI, auto-revue contre ce fichier + DESIGN.md
   du produit avant livraison.

## R109.5 — Violations

- Appliquer le même skin à deux produits sans décision R109.1 documentée → VIOLATION.
- Composant consommant une primitive ou valeur brute → VIOLATION.
- Thème dark livré sans re-check contraste des paires sémantiques → VIOLATION.
- UI générée sans consultation R109.3 quand le pattern existe → VIOLATION.

---
**Créée** : 2026-08-22 — issue recherche web (sources R109.3)
**Applies to** : tous projets lambda-Section
**Enforcement** : MANDATORY
