# PAPER_NUMBER_AUDIT.md — Audit des chiffres du papier arXiv contre les artefacts (Août 2026)

> MID: REPRO-007
> Owner: LambdaSection
> Status: DONE (audit initial 10/08 ; addendum canonique 13/08 — chiffres opposables avant soumission)
> Last updated: 2026-08-13
> Périmètre : `docs/paper_draft.md` + `docs/paper.tex` (draft v4 → v5 en préparation).
> Méthode : régénération des artefacts canoniques (2026-08-10, CPU, seed 42 quand déterministe)
> puis comparaison claim-par-claim. Règle : le papier ne cite QUE des nombres reproduits par
> un artefact dans le repo (Mom-test, pas de surestimation).

---

## 1. Résumé exécutif

| Claim papier (v4) | Artefact frais (2026-08-10) | Verdict |
|---|---|---|
| Sweep « 200 architectures, 6 familles » (MLP 50/CNN 40/RNN 40/TF 40/Hybrid 30) | 200 configs, 6 familles — MLP **38**/CNN **33**/RNN **33**/TF **33**/Hybrid **30**/BlackSwan **33** (`combinatorial_results.json`) | ✅ structure confirmée, ⚠ comptes corrigés |
| « 92% (277/300) » | **94.1% = 1 129/1 200** (245 s CPU) | ❌ remplacé (aucun artefact pour 277/300) |
| Familles v1.5.0 : MLP 93/CNN 90/RNN 71/TF 91/**Hybrid 96** | **100/96/73/100/100** (BlackSwan 94) — après fix générateur §2.1b | ❌ Hybrid 96 → 40 puis **100** (180/180) ; le « 40 » était un bug de shape du générateur |
| Fuzzer « 47/50 crash » | **4/20 crash (20%)**, 13 détectés, 3 ok (`fuzz_report.json`, seed 42) | ❌ remplacé (aucun artefact 50 iters) |
| Stress 15/15 | 15/15 (`stress_test_suite.py`) | ✅ confirmé |
| Benchmark 1/6 · 5/6 · 150 chaînes | identique, déterministe (`benchmark_honest.json`) | ✅ confirmé |
| OOS : md « ResNet 6/6, 5 events healthy » ; tex « 4 archis 100% » | **21/24 (88%)** après fix builder Mamba (voir §2.5b) : ResNet 6/6, ViT-Tiny 6/6, EfficientNet-B0 6/6, **Mamba-Mini 3/6** (0 crash ; 2 NO = scénarios no-op) — gate ≥90% FAIL | ⚠ md OK mais incomplet ; tex faux (Mamba ≠ 100%) ; « 0/6 CRASH composite » requalifié en bug builder |
| Healthy FP « <2 events » (abstract) | ResNet healthy **5** events ; ViT-Tiny healthy **127** (activation_regime_shift) ; EfficientNet 5 | ❌ « <2 » non soutenu → reformulé |
| Tiers black-swan 104/108 · 102/108 · RAG 36/36 · RL 0/36 | inchangés (`blackswan_{,tier2,tier4}_results.json`) | ✅ confirmés |
| Qwen v5 « 93.7%, 108 ex, 37 min » | rejoué 2026-08-11 (GPU, adapter commité `checkpoints_v5/final`, greedy, chat template) : **13.9% (15/108)** — collapse catégoriel (85× `disparition_de_gradient` + 16× `overfitting` inventé) ; **93.7% non reproductible** (§2.7) | ❌ claim retiré / qualifié |

## 2. Détail par artefact (runs frais 2026-08-10)

### 2.1 Combinatorial (`validate_combinatorial.py --full`, 245 s, `combinatorial_results.json`)
- 200 configs × 6 bugs = 1 200 évaluations ; overall **1 129/1 200 (94.1%)**.
- Familles : MLP 228/228 (100%) · CNN 191/198 (96.5%) · RNN 146/198 (73.7%) ·
  Transformer 198/198 (100%) · Hybrid **180/180 (100%)** · BlackSwan 186/198 (94%) — après fix §2.1b.
- Par bug : exploding 100%, vanishing 98%, zero_init 83%, nan_data 88%, dead_bias 94%, divergence 100%.
- BlackSwan du sweep : GNN 6/6, MoE 6/6, Diffusion 6/6, **RL 6/6** (note : le toy RL tier-4 reste 0/36 —
  générateur d'archis différent, signal via stats d'activation vs logprobs), RAG 6/6,
  NeuralODE 6/6, **FlashAttn 2/6** (3 configs `BS_FlashAttn_*`, inject. via SDPA).
- Divers : quelques configs dupliquées dans la séquence (générateur : 200 items, nommage pas 100%
  unique — sans impact sur les comptes par famille).

### 2.1b Hybrides RNN-composites requalifiés (post-audit, 2026-08-10) — bug générateur, PAS limite de détection

Le verdict initial « Hybrid 40% : hooks leaf silencieux sur modules entièrement composites
(`nn.ModuleList` imbriqués + RNN) » était **faux**. Reproduction isolée : dans `build_hybrid`,
le forward LSTM faisait `x = x.transpose(0,1)` puis `x = out[0]` (1er pas de temps) sur un LSTM
`batch_first=True` — incohérence de dimensions who écrasait la dimension batch : le modèle
**crashait au step 0** (`mat1 (1x16) and (64x10)`) sur les 18 configs RNN-composites
(`rnn+mlp`, `cnn+rnn+mlp`, `all` × 6 widths/activations), y compris la baseline saine. Le
`try/except → break` de `train_with_dbg` avalait le crash : 0 événements → « non détecté ».

- Fix : `x = out.mean(dim=1)` (batch_first=True, agrégation temporelle) + tail `fc` gérant le 2D.
- Re-run (245 s) : Hybrid **180/180 (100%)** — les hooks feuilles fonctionnent parfaitement dans
  les ModuleList imbriqués ; overall 1 129/1 200 (94.1%) ; par-bug 100/98/83/88/94/100.
- Conséquence : la limite « modules composites invisibles » (§8.2 item 4 du papier) n'est
  **plus soutenue par aucun artefact** — ni par Mamba (§2.5b), ni par le sweep. Elle est supprimée
  du papier. Les seules limites réelles restent : RNN 73%, FlashAttention in-sweep 2/6, GNN 88%
  (tiers), ViT healthy 127 events.

### 2.2 Fuzzer (`arch_fuzzer.py --runs 20`, seed 42, `fuzz_report.json`)
- 20 runs : **4 crash (20%)** `invalid_spec_after_5_retries` (bug « ? »), 13 détectés, 3 ok.
- Détection sur les buggy construits : 13/16 (81%) ; gaps : vanishing 0/1, mixed_precision 1/2, zero_init 1/2.

### 2.3 Stress (`stress_test_suite.py`) → 15/15 (100%), déterminé.

### 2.4 Benchmark (`benchmark_honest.py`, seed 42) → detect_anomaly 1/6 · W&B 5/6 · NeuralDBG 5/6 · 150 chaînes.

### 2.5 OOS v2 (`validate_oos.py`, 2026-08-10, `oos_validation_report.json`)
- **18/24 (75%)** au run initial ; gate ≥90% **FAIL** ; crash 6/24.
- ResNet-18 : 6/6 (events scénarios : 5, 245, 7, 54, 103, 352 ; healthy 5 = activation_regime_shift).
- ViT-Tiny : 6/6 (healthy **127 events** — sur-détection visible, bénin, à discuter).
- EfficientNet-B0 : 6/6 (healthy 5).
- **Mamba-Mini : 0/6 CRASH** (0 event, 0 chaîne — SSM custom 100% composite, hooks silencieux + crash).
- **2 bugs corrigés dans `validate_oos.py`** (découverts par l'audit) :
  1. `detected in ("YES","CRASH")` comptait les crashs comme détectés → gate faussé (24/24 affiché).
  2. `UnicodeEncodeError` sur ⚠/✅ en console cp1252 → crash avant l'écriture du rapport
     (fix : `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`).
  - + métadonnées du rapport rendues dynamiques (date 2026-07-08 codée en dur → date du run ;
    architecture décrite).

### 2.5b Mamba-Mini requalifié (post-audit, 2026-08-10) — bug builder, PAS limite de détection

Le verdict « Mamba 0/6 : SSM composite invisible aux hooks » était **faux**. Cause racine trouvée en
reproduisant le crash isolément : `torch.silu` (alias supprimé dans torch ≥ 2.13) → `AttributeError`
au step 0 de chaque scénario (`crash_error` de tous les runs Mamba du rapport initial :
`module 'torch' has no attribute 'silu'`). Le modèle n'atteignait jamais les hooks.

- Fix : `torch.silu` → `torch.nn.functional.silu` dans `validate_oos.py`.
- Re-run complet (`venv` torch 2.9.1 + torchvision 0.24.1, 175 s) → **21/24 (88%), 0 crash**,
  gate ≥90% FAIL (88% < 90%).
- **Mamba-Mini : 3/6** — exploding 63 events/30 chaînes · NaN data 20/30 · divergence 85/30 ;
  **healthy = 0 events** (baseline propre, mieux que ResNet/EffNet à 5) ;
  vanishing & zero_init = **NO car scénarios no-op** : les injections ciblent `layer3`(ReLU) et
  `layer4` (noms ResNet) qui n'existent pas dans Mamba → aucun bug réellement injecté.
- Conséquence : la limite « modules composites invisibles » n'est **plus soutenue par Mamba** ;
  elle reste soutenue par les 18 configs composites du sweep Hybrid (0/6, §2.1).

### 2.6 Tiers black-swan (non re-runs, artefacts existants cohérents)
- Tier 1 : 104/108 (GNN 88%, MoE 100%, Diffusion 100%) · Tier 2 : 102/108 (FA 100%, NODE 100%,
  Quantized 83%) · Tier 4 : RAG 36/36, RL 0/36 · Federated : 18/36.

### 2.7 Qwen v5 re-vérifié (2026-08-11, GPU Quadro M4000) — 93.7% NON reproductible

Méthode : adapter commité `Neural-Agent/neuralagent/model/checkpoints_v5/final` (commit b595495,
Jul 5, seul commit du modèle ; aucun changement non commité), base `Qwen/Qwen2-0.5B` (HF cache),
fp16, chat template (identique au format d'entraînement SFT), greedy, extraction `"categorie"`
+ fallback mots-clés — protocole = miroir de `neuralagent/model/predict.py` (température 0.15).
Inputs : les 108 exemples de `v5_training_data.json` (jeu d'ENTRAÎNEMENT, donc oracle favorable).

Résultat : **15/108 (13.9%)** — `qwen_v5_verify_results.json` (script `verify_v5_accuracy.py`
dans Neural-Agent). Collapse catégoriel : 85× `disparition_de_gradient`, 16× `overfitting`
(catégorie ABSENTE des 6 labels d'entraînement), 7× labels hallucinés. 3 OK exacts par famille
= les 3 vrais `disparition` (le modèle crache la catégorie majoritaire). Greedy et
température 0.15 cohérents (1/5 vs 1/5 sur la tranche).

Explication probable : les prompts des 108 exemples sont quasi-identiques entre bugs
(mêmes events `activation_regime_shift`/`gradient_health_transition` pour exploding vs vanishing
vs zero_init, cf. exemples 0-2 : seuls les comptes diffèrent) — signal de distinction minuscule,
le modèle a appris la distribution majoritaire, pas la distinction. Le « 93.7% » du changelog
n'a aucun artefact de mesure : trainer_state ne loggue qu'`eval_loss` (0.2285), aucune
`eval_accuracy`. Le v4 (92.3%) souffre de la même absence de mesure reproductible.

Verdict : claim retiré du papier ; le classifieur reste décrit comme assistance non chiffrée.

## 3. Actions papier prises (draft v5 + tex)

1. Abstract : « 92% (277/300) » → « 94% (1 129/1 200) » ; clause FP « <2 events » supprimée
   (remplacée par les faits ResNet 5 events).
2. §5.1/§5.2 : table 6 familles (+ comptes frais), tableau de détection frais + décomposition par bug,
   note honnête Hybrid 100% post-fix §2.1b / RNN 73%.
3. §5.6 : fuzzer 20 iters / 4 crash / 13 détectés (remplace « 47/50 »).
4. §5.13 : OOS v2 4 archis 21/24, tableau events par archi, Mamba 3/6 (0 crash, 2 no-op ResNet) après
   fix builder §2.5b.
5. §8.2 : limites (RNN tuple, ViT healthy 127 events) ; « composites invisibles » RETIRÉ (hybrides 100%
   post-fix, §2.1b).
6. §9/§10 : comptes coordonnés (200 sweep 94% + 108 tiers + OOS 4 archis 21/24 ; familles 6 + tiers).
7. Qwen v5 (abstract + §5.14) : « 93.7% per changelog, re-verification pending » → « re-vérifié
   2026-08-11 : 13.9% (15/108) sur le jeu d'entraînement, collapse catégoriel — précision retirée » (§2.7).

## 4. Suivis (avant soumission arXiv)

- [x] **Cause racine du crash Mamba-Mini élucidée (10/08)** : `torch.silu` supprimé dans torch 2.13 →
      `AttributeError` step 0. Fix `F.silu` + re-run 21/24 (0 crash). Pas de limite « SSM composite »
      confirmée : seul le sweep Hybrid (composites) la soutenait. Rapport `oos_validation_report.json`
      régénéré (175 s, venv torch 2.9.1).
- [x] **Hybrides composites : décision (10/08)** — le « 40% » était un bug de shape du générateur
      (forward LSTM `out[0]` vs `batch_first=True`), les 18 configs crashaient au step 0, jamais
      entraînées. Fix `out.mean(dim=1)` → Hybrid **100% (180/180)**, overall 94.1% (1 129/1 200).
      `register_composite_hook` : pas nécessaire pour ces architectures — la limite « composites
      invisibles » est retirée du papier (§2.1b).
- [x] **Re-vérification Qwen2-0.5B LoRA v5 (93.7%) sur GPU (11/08)** — ÉCHOUÉE : 13.9% (15/108,
      greedy, chat template, exemples d'entraînement) ; collapse catégoriel ; « overfitting » hors
      label. 93.7% non reproductible, retiré du papier (§2.7, `qwen_v5_verify_results.json`).
      Script : `Neural-Agent/verify_v5_accuracy.py`. Le v4 (92.3%) n'a pas plus d'artefact de mesure.
- [ ] Note : BS_RL 6/6 (sweep) vs RL 0/36 (tier-4 toy) — documenté au §5.10/§5.2 du papier
      (setup note, 10/08) ✅ (document envoyé avec le papier, pas de todo restant).
- [x] pipeline_report.json régénéré (10/08, passe `run_pipeline.py` complète) : status `oos_failed`
      honnête (21/24, 88% < gate 100) — + fixes pipeline : encodage utf-8 (même bug cp1252 que
      validate_oos), stage combinatorial passé en `--full` (le `--quick` écrasait l'artefact
      canonique 1,200 éval), rapport écrit même sur échec de gate (avant : rapport périmé 23/07).
      ⚠ A re-régénérer après le fix Hybrid (94%).

## 5. Addendum 2026-08-13 — chiffres canoniques (runs P2/P4/P5 + classifieur v6)

> Ces chiffres remplacent les runs 10/08 comme référence opposable. Fichiers : `combinatorial_results.json`
> (run 2026-08-13), `fuzz_report.json` (seed 42, déterministe), `benchmark_honest.json` (re-run 2026-08-13),
> `oos_validation_report.json` (injecteurs arch-agnostiques), `Neural-Agent/artifacts/classifier_v6/`.

### 5.1 Sweep combinatoire → 99,4 % (1 193/1 200)
- **RNN 67,7 % → 100 % (198/198)** : cause racine = `build_rnn` utilisait `out[:, -1, :]` sur LSTM
  bidirectionnel (dernier pas reverse = premier pas depuis h0=0 → gradient `W_hh_reverse` exactement nul ;
  baseline saine à 40 events « vanishing »). Fix `out.mean(dim=1)` + seuils famille adaptatifs
  (RNN +1, Hybrid/BlackSwan +2, autres +3, comparaison inclusive `>=`).
- Par bug : exploding 100 % · vanishing 100 % · dead_bias 100 % · divergence 100 % ·
  zero_init 98 % · nan_data 98,5 %.
- FlashAttention : `register_composite_hook` sur `nn.MultiheadAttention` → 9/18 → 12/18.
- **7 restants assumés (1× CNN gelu + 6× FlashAttn)** : masqués par l'heuristique moteur
  `saturation_ratio = (|x|>0.95)` absolu, inadaptée aux sorties Linear non bornées → baseline saine
  faussement « saturated ». **P2b = fix moteur requis** (saturation relative ou conditionnée à
  l'activation) + re-validation complète. Le papier DOIT citer 99,4 % avec cette limite divulguée,
  pas 100 %.

### 5.2 Fuzzer → 0 crash, 19/19 sur bugs manifestés
- Les 4 crashs étaient un bug du générateur de specs (`input_dim` aléatoire vs build fixe 16).
  Machine d'états de formes + `min_trainable=2` → 0 crash. RNN seq_len=1 → T=4 (même classe
  d'artefact h0=0 que §5.1).
- **1 run « ok » assumé (seed 51)** : injection asymptomatique (LayerNorm absorbe l'échelle fp16).
  Ne pas forcer une détection = refuser de fabriquer un signal.
- Le papier NE DOIT PLUS citer « 47/50 (94 %) crash » (aucun artefact 50 iters).

### 5.3 Benchmark → 5/5 + 0 FP + 150 chaînes
- Le « 5/6, 1 scénario raté » était un **artefact de scoring** : le scénario raté était le contrôle
  sain (silence = comportement correct). Nouveau scoring : bugs injectés **5/5**,
  contrôle sain en gate FP séparée (`false_positives_healthy`).
- Résultat opposable : **NeuralDBG 5/5 + 0 FP + 150 chaînes ; W&B 5/5 (0 chaîne) ; detect_anomaly 1/5.**
- Le papier DOIT utiliser le dénominateur 5 (pas 6) et reporter le contrôle sain séparément.

### 5.4 OOS → 24/24 (100 %), baselines 1/2/0/0
- ResNet-18 6/6 · ViT-Tiny 6/6 · EfficientNet-B0 6/6 · Mamba-Mini 6/6 (injecteurs arch-agnostiques ;
  0 crash, 0 FP — healthy baselines : 1/2/0/0 events). Commande : `python validate_oos.py`.
- Historique (ne pas citer comme résultat) : « Mamba 0/6 crash » = bug builder `torch.silu` ;
  run 21/24 = injecteurs ResNet-only no-op ailleurs.

### 5.5 Classifieur v6 — seule description chiffrée autorisée
- SFT v5/v6 : ÉCHEC (13,9 % / 0,71 %) — fine-tuning génératif ≠ classification ; conserver comme
  preuve (`checkpoints_v6`), ne jamais citer comme succès.
- **Solution livrée** : classifieur linéaire sur embeddings Qwen2-0.5B GELÉS (dernier token 896-d) —
  train 281/281, **holdout 29/29 (100 %)**, prompts v6 discriminants.
- Phrasé papier autorisé : « linear probe on frozen Qwen2-0.5B embeddings, 29/29 holdout ».
  INTERDIT : « fine-tuned Qwen2-0.5B LoRA classifier achieves 93.7 % » (changelog retiré, §2.7).

### 5.6 PRs upstream — phrasé autorisé (zéro claim mergé)
- État : 2 PRs test ouvertes (#188053 svdvals, #188923 santé gradient), 2 closes/retirées
  (#188933 fix, #188066 test — prémisse réfutée par albanD sur #184575), **0 mergée**.
- Leçon process (à citer) : discuter sur l'issue + label *actionable* AVANT d'ouvrir une PR.
- « Six real PyTorch bugs discovered » → reformuler « reproduced and diagnosed »
  (ex. BUG-007 signalé par @ezyang). BUG-008 (F.normalize) NE DOIT PAS figurer comme succès.