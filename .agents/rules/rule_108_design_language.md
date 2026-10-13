---
name: rule-108-design-language
description: 'RULE 108: Design Language — « Quiet Precision » (LifeTrack & dérivés)'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_108_design_language.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 108: Design Language — « Quiet Precision » (LifeTrack & dérivés)
**Statut : MANDATORY. Référence complète : `DESIGN.md` + `DESIGN_SYSTEM.md` (racine du repo).**

## R108.0 — Gate Impeccable obligatoire (directive CEO 2026-08-21 — CONSERVÉE)
Toute surface UI générée ou modifiée passe le détecteur **Impeccable** (`npx -y impeccable detect`) :
1. LifeTrack et dérivés : `npx -y impeccable detect src/` → **exit 0** requis avant chaque milestone visuel (R58).
2. CI : step `impeccable detect` dans truth-daily.yml (rapport `ci-status.json`).
3. Contexte design vivant : `DESIGN.md` à la racine = source de vérité tokens/voice. Toute exception y est documentée, sinon rejetée.
4. Re-vérifier un concurrent supérieur tous les 6 mois (prochaine échéance : 2027-02).

## R108.1 — Identité (v2, 2026-08-22 : remplace Ledger Brutal)
« Quiet Precision » : neutre chaleureux, dense mais respirant. Radius 6px,
ombres douces en blur, bordures 1px/2px structurelles. La couleur = donnée
uniquement. Le thème Noir & Blanc reste disponible comme thème à contraste
maximal — adouci (radius hérité, ombres douces, sans uppercase forcé ni
texture halftone).

## R108.2 — Interdictions absolues (« anti-AI slop »)
1. Aucun gradient décoratif, glassmorphism, backdrop-blur, glow, néon.
2. **Aucune border-left épaisse "side-tab"** (tell #1 détecté par Impeccable) →
   barre interne via `box-shadow: inset`.
3. Pas d'animations width/height (layout thrash) → transform/opacity.
4. Pas de bounce/elastic easing → ease-out quart/expo.
5. Aucun nombre affiché sans contexte (n/p/fenêtre).
6. Langue de l'UI = langue de l'utilisateur.

## R108.3 — Composants signatures
Cartes à barre interne inset · zébrure subtile · mini-calendrier bimodal
(action/vigilance) · directive du jour · chips médailles · summary pliables.

## R108.4 — Motion
`cubic-bezier(0.2,0,0,1)` ou ease-out-quart, 150–250ms. Pulse réservé streak ≥3.
