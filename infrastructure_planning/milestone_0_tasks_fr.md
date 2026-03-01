# Tâches DevOps / MLOps — Jalon 0 (Configuration Initiale)

---

## 🧠 Brief Pédagogique pour le CEO (Pourquoi Déléguer ?)

**Q: "Est-ce que je peux tout faire moi-même ? Ça vaut la peine de déléguer si ça prend un mois ?"**

**Oui, vous *pouvez* le faire vous-même.** Techniquement, écrire une GitHub Action prend quelques heures. 
**CEPENDANT :** Chaque changement de contexte ("context switch") de "CEO/Chercheur ML" à "Plombier DevOps" draine votre charge cognitive. Si vous passez 20 heures à réparer un bug de volume Docker ou un pipeline CI, ce sont 20 heures que vous n'avez PAS passées à parler aux utilisateurs (Mom Test) ou à améliorer votre algorithme ML central. Déléguer vous permet de rester dans le "Moyeu" (Core Logic) pendant que votre ami construit les "Rayons" (Delivery Pipeline).

### Le concept de "Porte" CI/CD

**Q: "Si la porte n'est pas dans le cloud, c'est facile de commit du code qui ne respecte pas les critères ?"**

**Exactement.** Pour le moment, vous avez des outils locaux (`pytest`, `bandit`), mais ils reposent sur une *discipline humaine*. Un développeur fatigué peut facilement taper `git push` sans lancer les tests. Un Pipeline CI/CD Cloud est un **videur de sécurité numérique incassable**.

```text
  [FLUX LOCAL - Vulnérable à l'erreur humaine]
  
  Machine du Dev        Dépôt Git
  +---------------+    +--------+
  | Mauvais Code  | -> | MERGED | (Oups! Code cassé dans master)
  +---------------+    +--------+
         push          ^
                       Ont-ils lancé les tests ? 
                       Peut-être. Peut-être pas.

  [FLUX CI/CD CLOUD - Hub Incassable]
  
  Machine du Dev                     GitHub / Cloud
  +---------------+    +-------------------------------------+      +--------+
  | Mauvais Code  | -> | [LE VIDEUR CI]                      |      |        |
  +---------------+    | 1. Tests (Couverture < 60%)      ❌ | -/-> | MERGED |
         push          | 2. Securité (Bandit échoue)      ❌ |      |        |
                       +-------------------------------------+      +--------+
                                         |
                                         v
                            [PUSH REJETÉ - RÉESSAYEZ]
```

---

## 🛠️ Les 5 Tâches

### 1. Intégration Continue (CI/CD) Multiplateforme avec Portes de Validation
* **Tâche**: Créer un workflow GitHub Actions qui lance automatiquement les tests sous Linux et Windows. Ce pipeline doit bloquer tout code qui ne respecte pas la Règle 5 (60% de couverture mini) et la Règle 6 (scans de sécurité `bandit`/`safety`).
* **Gain Estimé**: ~2h/semaine de tests manuels sauvées. Élimine le risque de casser le projet avec du mauvais code.

### 2. Tracking d'Expériences et de Modèles (MLOps)
* **Tâche**: Intégrer un tracker (MLflow ou W&B) dans `demo_vanishing_gradients.py` et les futurs scripts. Suivre l'apprentissage, les gradients et les paramètres automatiquement plutôt que d'afficher des graphiques manuellement.
* **Gain Estimé**: ~3h par itération de modèle passées à éplucher des logs textes.

### 3. Espaces de Travail Hermétiques via Docker (DevOps)
* **Tâche**: Créer un `Dockerfile` et un `docker-compose.yml` propres à PyTorch. Inclure les "volumes" pour la data afin de développer localement sans conflits de paquets Python.
* **Gain Estimé**: Sauve 4 à 5h d'installation (onboarding) pour chaque nouvelle personne ou IA rejoignant le projet.

### 4. Versioning des Données et Binaires (MLOps)
* **Tâche**: Mettre en place DVC (Data Version Control) pour gérer les images (ex: `synthetic_data_sample.png`) et les futurs poids de modèles. Les retirer de Git.
* **Gain Estimé**: Empêche le dépôt Git de peser 10 Go. Sauve des minutes d'attente à chaque upload/download.

### 5. Script d'Automatisation de Synthèse (DX / DevOps)
* **Tâche**: Créer un script (ex: Python avec `python-docx`) qui convertit automatiquement `SESSION_SUMMARY.md` en fichier `.docx` ou PDF propre pour la communication externe.
* **Gain Estimé**: Sauve au lead developer ~15 minutes par session, soit ~2h/semaine de travail administratif.
