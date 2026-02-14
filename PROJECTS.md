# Roadmap — Projets A & B

> Un seul projet à la fois. Développés ici.

---

## Projet A — Transformer Probabiliste pour Séries Temporelles

**Orientation**: OpenQuant

### Objectif
Modéliser une **distribution conditionnelle** : \( P(Y_{t+1} \mid X_{1:t}) \). Pas une valeur. Une **incertitude exploitable**.

### Architecture V1
- Input embedding (features + time encoding)
- Positional encoding
- Transformer Encoder
- Head probabiliste : μ (mean), σ (std), optionnel mixture logits

### Dataset
- **Commencer synthétique** (sinusoïde bruitée)
- Plus tard : Crypto OHLCV, Forex

### Loss
NLL Gaussian : \( L = \frac{1}{2}\log(\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2} \)

### Extensions futures
- Multi-head temporal attention, multi-horizon, calibration (ECE), backtesting

---

## Projet B — Adaptive Gradient Architecture

**Orientation**: Neural / NeuralDBG

### Objectif
Observer et corriger la dynamique interne. Couche qui mesure ‖∇W‖, EMA, rescale si gradient < seuil.

### Structure
```python
class AdaptiveGradientWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.grad_ema = None
```

### Mécanisme
Si ‖∇W‖ < α · EMA → ∇W ← β · ∇W (β > 1)

### Expériences
MLP 50 couches, RNN long, Transformer profond. Comparer convergence, stabilité, distribution des gradients.
