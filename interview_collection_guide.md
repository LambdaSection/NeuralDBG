# Guide de Collecte d'Interviews — NeuralDBG Mom Test

## Francais

### Follow-up MechaAthro (Interview #2)

**Contexte**: Reponse courte "MLFlow est solide" - signal neutre. Necessite creusage.

**Questions de suivi (a poser sur Discord)**:

1. "Tu mentionnes MLFlow - tu l'utilises pour quoi exactement ? Tracking de metriques, ou aussi pour debugger ?"
2. "La derniere fois que t'as eu un training qui part en vrille (loss qui explose, gradients qui vanish, etc.), t'as fait quoi ?"
3. "Combien de temps t'as passe a comprendre ce qui se passait ?"
4. "Est-ce qu'il t'est arrive de ne pas comprendre pourquoi un training echouait ?"
5. "T'as deja abandonne un projet/model a cause d'un probleme d'entrainement incomprehensible ?"

**Signaux a surveiller**:
- **Positif**: "J'ai passe X jours", "J'ai du faire un script custom", "J'ai abandonne"
- **Negatif**: "MLFlow me suffit", "Ca m'arrive rarement"
- **Neutre**: Reponses vagues, "Ca depend"

---

### Ressources pour Interviews Supplementaires

#### Subreddits a Explorer
1. **r/MachineLearning** — Questions de recherche ML
2. **r/deeplearning** — Questions techniques deep learning
3. **r/pytorch** — Problemes specifiques PyTorch
4. **r/tensorflow** — Problemes specifiques TensorFlow
5. **r/learnmachinelearning** — Debutants avec problemes frequents

#### Mots-cles de Recherche Reddit
```
site:reddit.com "training divergence" "why"
site:reddit.com "vanishing gradients" debug
site:reddit.com "loss spike" "what caused"
site:reddit.com "training instability" help
site:reddit.com "NaN loss" deep learning
site:reddit.com "model not learning" why
```

#### Questions a Poser (Mom Test Compliant)
1. "Racontez-moi la derniere fois que votre entraînement a échoué inopinément."
2. "Combien de temps avez-vous passé à comprendre pourquoi ?"
3. "Qu'avez-vous fait concrètement pour debugger ?"
4. "Avez-vous déjà cherché ou construit une solution ?"
5. "À quelle fréquence ce problème se produit-il ?"

---

### Discord Servers a Explorer
1. **PyTorch Discord** — Channels #help, #questions
2. **FastAI Discord** — Channel #debugging
3. **Hugging Face Discord** — Channel #training-issues
4. **r/MachineLearning Discord**

---

## English

### Resources for Additional Interviews

#### Subreddits to Explore
1. **r/MachineLearning** — ML research questions
2. **r/deeplearning** — Technical deep learning questions
3. **r/pytorch** — PyTorch specific issues
4. **r/tensorflow** — TensorFlow specific issues
5. **r/learnmachinelearning** — Beginners with frequent problems

#### Reddit Search Keywords
```
site:reddit.com "training divergence" "why"
site:reddit.com "vanishing gradients" debug
site:reddit.com "loss spike" "what caused"
site:reddit.com "training instability" help
site:reddit.com "NaN loss" deep learning
site:reddit.com "model not learning" why
```

#### Questions to Ask (Mom Test Compliant)
1. "Tell me about the last time your training failed unexpectedly."
2. "How much time did you spend figuring out why?"
3. "What did you concretely do to debug it?"
4. "Have you already searched for or built a solution?"
5. "How frequently does this problem occur?"

---

### Discord Servers to Explore
1. **PyTorch Discord** — Channels #help, #questions
2. **FastAI Discord** — Channel #debugging
3. **Hugging Face Discord** — Channel #training-issues
4. **r/MachineLearning Discord**

---

## Template de Post Discord/Reddit

### Version Francaise

```
Sujet: Question de recherche sur le debugging d'entraînement

Question serieuse de quelqu'un qui fait de la recherche en ML.

Quand un modèle s'effondre, diverge, ou se comporte bizarrement pendant l'entraînement 
(pas des erreurs de syntaxe, mais des problèmes de dynamique d'entraînement) :

• gradients qui explosent / disparaissent
• pics de loss soudains
• neurones morts
• instabilité tardive
• comportement dépendant du seed

Comment faites-vous habituellement pour comprendre *pourquoi* ?

Comptez-vous sur TensorBoard / W&B ? Ajoutez des hooks et print les tensors ? 
Re-lancez avec différents hyperparamètres ? Simplifiez le modèle ?

Je n'ai pas besoin de "best practices", j'essaie de comprendre ce que les gens 
font *réellement* aujourd'hui et ce qui semble le plus douloureux.

Merci pour vos retours !
```

### Version Anglaise

```
Subject: Research question on training debugging

Serious question from someone doing ML research.

When a model suddenly diverges, collapses, or behaves strangely during training 
(not syntax errors, but training dynamics issues):

• exploding / vanishing gradients
• sudden loss spikes
• dead neurons
• instability that appears late
• behavior that depends on seed or batch order

How do you usually figure out *why* it happened?

Do you rely on TensorBoard / W&B metrics? Add hooks and print tensors? 
Re-run experiments with different hyperparameters? Simplify the model and hope it goes away?

I'm not asking for best practices, I'm trying to understand what people *actually do* today,
and what feels most painful or opaque in that process.

Thanks for your feedback!
```

---

## Tracking des Interviews

| # | Source | Date | Signal | Status |
|---|--------|------|--------|--------|
| 1 | Reddit r/neuralnetworks | 2026-02-23 | POSITIF | Complete |
| 2 | | | | A collecter |
| 3 | | | | A collecter |
| 4 | | | | A collecter |
| 5 | | | | A collecter |

---

## Prochaines Etapes

1. [ ] Poster sur r/MachineLearning avec le template
2. [ ] Poster sur r/pytorch avec le template
3. [ ] Rejoindre PyTorch Discord et poser la question
4. [ ] Documenter chaque reponse dans mom_test_results.md
5. [ ] Analyser les signaux apres 5 interviews
6. [ ] Prendre decision GO/NO-GO/PIVOT