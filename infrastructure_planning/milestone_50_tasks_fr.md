# Taches DevOps/MLOps - Jalon 50%

Au stade de progression de 50 %, focus sur le suivi des experiences et le versionnage des donnees.

## Taches

1. **Integration du Logging d'Evenements MLflow**
   - **Description** : Integrer MLflow pour loguer les objets `SemanticEvent` et les hypotheses causales automatiquement.
   - **Criteres d'acceptation** : La boucle d'entrainement logue les evenements sur un serveur MLflow local ou distant.
   - **ROI** : Economise 1 jour par session d'analyse en fournissant des historiques persistants.
   - **Ticket Linear** : `infra/MLO-6-mlflow-tracking`

2. **Versionnage des Donnees (DVC) pour Donnees Synthetiques**
   - **Description** : Configurer DVC pour suivre les jeux de donnees synthetiques utilises pour les demos de debug.
   - **Criteres d'acceptation** : `dvc push` et `dvc pull` fonctionnent pour les ondes generees.
   - **ROI** : Garantit une reproductibilite parfaite et economise des heures de regeneration de donnees.
   - **Ticket Linear** : `infra/MLO-7-dvc-setup`

3. **Generation Automatique de Documentation (Sphinx/MkDocs)**
   - **Description** : Configurer la generation automatique de la documentation API a partir des docstrings.
   - **Criteres d'acceptation** : La documentation se build sans erreurs et couvre toutes les methodes principales de `NeuralDbg`.
   - **ROI** : Economise 4 heures par mois de mise a jour manuelle de la doc.
   - **Ticket Linear** : `infra/MLO-11-auto-docs`

4. **Integration du Profilage des Ressources**
   - **Description** : Ajouter un profilage leger des ressources (memoire/GPU) aux evenements semantiques.
   - **Criteres d'acceptation** : Les evenements capturent les pics de memoire ou les baisses d'utilisation du GPU pendant les echecs.
   - **ROI** : Identifie les goulots d'etranglement 50 % plus rapidement.
   - **Ticket Linear** : `infra/MLO-12-resource-profiling`

5. **Renforcement des Scans de Securite (Safety Check)**
   - **Description** : Integrer `safety` dans la CI/CD et le pre-commit pour verifier les dependances vulnerables.
   - **Criteres d'acceptation** : La CI echoue si une vulnerabilite de haute severite est detectee.
   - **ROI** : Conformite de securite critique (Regle 6).
   - **Ticket Linear** : `infra/MLO-13-safety-check-integration`
