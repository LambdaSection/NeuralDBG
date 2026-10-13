---
name: rule-111-finance-local
description: 'RULE 111: Local Finance Data — données financières 100% locales — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_111_finance_local.md — DO NOT EDIT. scope=company priority=2 status=active -->

## RULE 111: Local Finance Data — données financières 100% locales — MANDATORY

### Rule

Les données financières de l'entreprise lambda-Section (trésorerie, dépenses,
revenus, burn rate, runway, MRR, CAC/LTV) ne quittent JAMAIS cette machine.

- Elles vivent uniquement dans `finances.local.json` (racine du repo, **gitigné**).
- Le seul fichier committable est le template `finances.local.example.json`
  (chiffres factices).
- Interdictions absolues :
  - inclure ces données dans un prompt LLM ou une requête vers une API distante ;
  - les committer, les pousser, ou les coller dans une issue / PR / Discord / X ;
  - les copier dans `dashboard-data.json`, `SESSION_SUMMARY.md`, `KURO_ACTIONS_LOG.md`.
- Lecture/écriture des calculs : `scripts/kuro_finance.py` uniquement (zéro réseau).
- Historique des relevés : `~/.kuro/finance_history.jsonl` (local uniquement,
  hors repo, jamais commité).
- Exposition autorisée : endpoint local `/api/finance` de `scripts/kuro_api.py`
  (bind 127.0.0.1), panneau KuroPulse local et desk web local.

Format de `finances.local.json` :

```json
{
  "currency": "USD",
  "starting_cash": 25,
  "months": [
    {
      "month": "2026-05",
      "expenses": [{ "label": "...", "amount": 25 }],
      "revenues": []
    }
  ],
  "acquisition": {
    "monthly_marketing_spend": 30,
    "new_customers_per_month": 3,
    "arpu_monthly": 15,
    "gross_margin_pct": 80,
    "avg_customer_lifetime_months": 12
  }
}
```

La clé `acquisition` est optionnelle ; elle alimente les unit economics :
CAC = spend ÷ nouveaux clients, LTV = ARPU × marge × durée de vie,
ratio LTV:CAC (sain ≥ 3).

Calculs produits par `kuro_finance.py` :

| Métrique     | Définition                                          |
| ------------ | --------------------------------------------------- |
| Burn mensuel | moyenne des dépenses sur la fenêtre analysée        |
| MRR          | moyenne des revenus mensuels                        |
| Net burn     | burn − MRR                                          |
| Runway       | starting_cash ÷ net burn (en mois), sinon « infini » |
| Statut       | critical < 3 mois ≤ warning < 6 mois ≤ healthy      |

### Verification

```
ACTION: après toute manipulation de finances.local.json, vérifier :
VERIFY: 1. `git status --porcelain` ne liste PAS finances.local.json
        2. aucun appel réseau dans le code touché (grep requests/urllib/WebClient sortant)
```

### Enforcement

IF une étape de travail doit exposer les chiffres financiers hors machine :

- STOP immédiat
- proposer une version agrégée SANS montants (ex. « runway < 3 mois »)
- demander confirmation explicite à l'utilisateur

IF finances.local.json est absent :

- copier `finances.local.example.json` puis laisser l'utilisateur remplir ses chiffres

---
