---
name: rule-104-auto-issues-tracking
description: 'RULE 104: Auto-Issues & Tracking — Création Obligatoire d''Issues pour
  Chaque Action'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_104_auto_issues_tracking.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 104: Auto-Issues & Tracking — Création Obligatoire d'Issues pour Chaque Action

## Problème
L'IA travaille sur des tâches complexes (déploiement, code, R&D) sans créer d'issues GitHub/Linear pour tracer chaque étape. Résultat : pas de visibilité, pas d'historique, les actions ne sont pas reliées à la roadmap.

## Solution
L'IA DOIT créer des issues (GitHub Issues ou Linear) pour TOUTE action technique avant de commencer, et commenter chaque étape au fur et à mesure.

## Règle

### 1. Déclencheur Obligatoire
À chaque début de session ou changement de tâche significatif :
- [ ] Vérifier si une issue existe déjà pour cette tâche
- [ ] Si NON : créer l'issue immédiatement
- [ ] Si OUI : référencer l'issue dans la session

### 2. Format Standard d'une Issue

```markdown
## Phase X: [Nom de la tâche]

**Projet :** [DevDemeterDAO | Constant_Yield | XP_Farming_System | GALT-Flatcoin-Concept]
**Priorité :** [Haute | Moyenne | Basse]
**Dépend de :** #[issue-id]

### Objectif
Une phrase expliquant le but.

### Checklist des Étapes
- [ ] Étape 1 : [description]
- [ ] Étape 2 : [description]
- [ ] Étape 3 : [description]

### Critères de Succès
- [ ] Résultat attendu 1
- [ ] Résultat attendu 2

### Notes
Liens vers les fichiers concernés, la doc, etc.
```

### 3. Commenter Chaque Étape
Après chaque micro-tâche accomplie :
- Ajouter un commentaire sur l'issue : `✅ Étape X terminée : [description de ce qui a été fait]`
- Si bloqué : `❌ Bloqué sur étape X : [raison du blocage]`

### 4. Mapping Roadmap → Issues
Toute phase dans DEV_ROADMAP.md doit avoir des issues correspondantes :
- Chaque Phase → Epic ou Milestone
- Chaque Tâche dans la phase → Issue individuelle avec checklist
- Les dépendances entre phases → liens "blocks/blocked by"

### 5. Fin de Session
Avant de terminer une session :
- [ ] Toutes les issues des tâches commencées sont en "In Progress"
- [ ] Toutes les issues des tâches terminées sont en "Done"
- [ ] Un commentaire récapitule ce qui a été fait
- [ ] Les nouvelles issues découvertes sont créées en "Backlog"

## Vérification

```
Au début de chaque session :
  IF aucune issue ouverte pour la tâche courante :
    STOP → créer l'issue avant de coder

Pendant le travail :
  Après chaque micro-tâche → commenter l'issue

En fin de session :
  IF issues non à jour :
    STOP → mettre à jour avant de terminer
```

## Exemple Concret

**Tâche :** Déploiement Base Sepolia

Issue créée automatiquement :
```markdown
## Phase 0: Déploiement gouvernance sur Base Sepolia

**Projet :** DevDemeterDAO
**Priorité :** Haute

### Checklist
- [ ] Créer un wallet de test
- [ ] Configurer le .env
- [ ] Récupérer des ETH faucet
- [ ] Lancer deploy-refactored.js
- [ ] Vérifier les contrats déployés
```

Puis chaque étape est commentée :
- ✅ Wallet créé : `0x9569...`
- ✅ .env configuré
- ❌ Bloqué : besoin d'ETH faucet — expliquer ce qu'est un faucet à l'utilisateur
- etc.

---

**Créée :** 2026-05-20
**Déclencheur :** Session de déploiement sans issues ni suivi visible
**Application :** MANDATORY — tous projets DemeterDAO
