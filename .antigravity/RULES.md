# AntiGravity Rules — NeuralDBG / OpenQuant Roadmap

> **Workflow**: On travaille sur **un seul projet à la fois**. Finir A avant de passer à B, ou vice versa. Pas de développement en parallèle.

---

## 🔷 Projet A — Transformer Probabiliste pour Séries Temporelles

**Orientation**: OpenQuant

### Objectif
Modéliser une **distribution conditionnelle** :  
\( P(Y_{t+1} \mid X_{1:t}) \)  

Pas une valeur. Une **incertitude exploitable**.

### Architecture V1 (minimaliste mais sérieuse)
- Input embedding (features + time encoding)
- Positional encoding
- Transformer Encoder
- Head probabiliste :
  - μ (mean)
  - σ (std)
  - Optionnel : mixture logits

### Dataset
- **Commencer synthétique** (sinusoïde bruitée)
- Si ça ne marche pas sur du propre, ça ne marchera pas sur du marché
- Plus tard : Crypto OHLCV, Forex

### Loss
Negative Log Likelihood (Gaussian) :
```
L = (1/2) log(σ²) + (y − μ)² / (2σ²)
```
Ça force le modèle à calibrer son incertitude.

### Extensions futures
- Multi-head temporal attention
- Multi-horizon forecasting
- Calibration testing (Expected Calibration Error)
- Backtesting avec gestion du risque

---

## 🔶 Projet B — Adaptive Gradient Architecture

**Orientation**: Neural / NeuralDBG

### Objectif
On ne prédit rien. On **observe et corrige** la dynamique interne.

Créer une couche qui :
- Mesure la norme des gradients layer-wise
- Détecte une décroissance anormale
- Applique une correction adaptative

### Concept V1 (simple et puissant)
À chaque backward pass :
1. Calculer ‖∇W‖
2. Maintenir une moyenne mobile (EMA)
3. Si gradient < seuil dynamique → rescale

### Structure cible
```python
class AdaptiveGradientWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.grad_ema = None
```
On encapsule n'importe quelle couche.

### Mécanisme
Si ‖∇W‖ < α · EMA  
Alors ∇W ← β · ∇W  (avec β > 1)

### Expériences à mener
- Tester sur : Deep MLP 50 couches, RNN long sequence, Transformer profond
- Comparer : convergence speed, stabilité, distribution des gradients

---

## Artifacts
Maintenir les artefacts générés sous `./.antigravity/artifacts/` (reports, plots, checkpoints, etc.).
