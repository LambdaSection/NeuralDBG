# Roadmap — Projets A & B

> Un seul projet à la fois.

---

## Projet A — Transformer Probabiliste pour Séries Temporelles

**Repo**: [Quant-Search](https://github.com/Quant-Search) (organisation) — repo dédié  
**Debug**: NeuralDBG (ce repo) — utilisé pour déboguer l'entraînement au fur et à mesure  
**Workflow**: Développer A dans Quant-Search, améliorer NeuralDBG et le projet A itérativement

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

### Setup repo Quant-Search
1. Créer un nouveau repo sous l'organisation Quant-Search (ex: `probabilistic-time-series`)
2. Copier le squelette : `skeleton-quant-search/` → contenu du nouveau repo
3. Dépendances : `pip install -e .` ; NeuralDBG en local : `pip install -e /path/to/NeuralDBG`
4. Lancer : `python -m src.train`
5. Itérer : améliorer le modèle dans Quant-Search, améliorer NeuralDBG selon les besoins de debug

---

## Projet B — Adaptive Gradient Architecture

**Repo**: NeuralDBG (ce repo)

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
