# Taches DevOps/MLOps - Jalon 25%

Au stade de progression de 25 %, les taches suivantes sont requises pour renforcer l'infrastructure et la fiabilite de NeuralDBG.

## Taches

1. **Pipeline CI Automatise (GitHub Actions)**
   - **Description** : Implementer un workflow GitHub Actions qui execute `pytest`, `bandit`, et `safety` a chaque push et PR.
   - **Criteres d'acceptation** : Le workflow passe sur la branche main, rapporte la couverture, et bloque les merges en cas d'echec de securite.
   - **ROI** : Economise 2 heures par semaine de verification manuelle.
   - **Ticket Linear** : `infra/MLO-4-github-actions-ci`

2. **Application des Hooks Pre-commit**
   - **Description** : Configurer et imposer des hooks pre-commit stricts pour tous les contributeurs.
   - **Criteres d'acceptation** : `.pre-commit-config.yaml` inclut `black`, `isort`, `flake8`, et `bandit`.
   - **ROI** : Previent 90 % des regressions de linting et de securite de base.
   - **Ticket Linear** : `infra/MLO-9-pre-commit-enforcement`

3. **Environnement de Test Hermetique (Docker)**
   - **Description** : Creer un `Dockerfile` pour un environnement de test standardise.
   - **Criteres d'acceptation** : `docker build` reussit et `pytest` s'execute dans le conteneur sans erreurs specifiques a l'environnement.
   - **ROI** : Reduit les problemes "ca marche sur ma machine" de 100 %.
   - **Ticket Linear** : `infra/MLO-5-docker-reproducibility`

4. **Documentation de la Politique de Securite**
   - **Description** : Creer un fichier `SECURITY.md` complet conformement a la regle 6.
   - **Criteres d'acceptation** : Le fichier existe avec des instructions de rapport claires et des exigences de scan automatise.
   - **ROI** : Critique pour les standards professionnels (Regle 6).
   - **Ticket Linear** : `infra/MLO-10-security-policy`

5. **Script de Verification de Build Windows**
   - **Description** : Creer un script PowerShell pour verifier le build sur Windows (Regle 17).
   - **Criteres d'acceptation** : Le script s'execute, build le projet, et execute la demo sans erreur.
   - **ROI** : Economise 1 jour de test manuel par release.
   - **Ticket Linear** : `infra/MLO-8-windows-packaging`
