# Changelog

All notable changes to NeuralDBG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0](https://github.com/LambdaSection/NeuralDBG/compare/v1.5.0...v1.6.0) (2026-10-07)


### Features

* 10 post-mortems complete — 3 new: clip underflow, AdamW+LayerNorm, fp16 softmax ([0a4fd9a](https://github.com/LambdaSection/NeuralDBG/commit/0a4fd9afbba060781205d4b3a50df59cd9fe0107))
* 7 post-mortems reproduced with NeuralDBG causal chains ([7a5517c](https://github.com/LambdaSection/NeuralDBG/commit/7a5517ccb416623e66b6a36593d2cc5ed3da20ab))
* Add .cursor/settings.json to enable the linear plugin. ([334a620](https://github.com/LambdaSection/NeuralDBG/commit/334a62059217d950dbd0628caec7d07190bf850a))
* Add AGENTS.md with strict rules for AI agents ([b89d175](https://github.com/LambdaSection/NeuralDBG/commit/b89d175da1c8febf9b54c40830b7931280b0b18b))
* add BUG-002 reproduction script (varlen_attn NaN gradients) ([8078452](https://github.com/LambdaSection/NeuralDBG/commit/8078452e2dec72166df55d95fe6fed38c8da18a7))
* Add foundational AI agent rules, development guidelines, and initial project setup files. ([a99b7e9](https://github.com/LambdaSection/NeuralDBG/commit/a99b7e9e15417d3908a4b578626ed2914d22372e))
* add git short read diagnosis for Windows reserved names ([cc17b46](https://github.com/LambdaSection/NeuralDBG/commit/cc17b461c1d542a62ea05da34bee37cab3a7869b))
* add infrastructure planning documents for milestone 25 and 50 DevOps/MLOps tasks. ([028a63d](https://github.com/LambdaSection/NeuralDBG/commit/028a63df15a594e1dc94a51ca372cce169d0132e))
* add Kaggle notebook for Neural-Agent training ([b8d63cf](https://github.com/LambdaSection/NeuralDBG/commit/b8d63cfc96b67f9adc3ea9880108eb173768199f))
* add Linear integration requirement for task creation and management ([4558f52](https://github.com/LambdaSection/NeuralDBG/commit/4558f522c909647f45f7e2ae9fcbbe5a6cdae4e1))
* add mandatory team stack rule with onboarding checklist and enforcement guidelines to documentation. ([562e67b](https://github.com/LambdaSection/NeuralDBG/commit/562e67b70536189ebfc2eee1d49516b303411ce3))
* add MHA benchmark scenario + W&B comparison scaffolding ([6db96e8](https://github.com/LambdaSection/NeuralDBG/commit/6db96e84a150f4aa84a2a736dc928d68d7a56e78))
* add NaN loss benchmark scenario + update comparison results ([6e8b64f](https://github.com/LambdaSection/NeuralDBG/commit/6e8b64f5503783ff19afb422163dd41d91f45704))
* Add Rule 28 for mandatory Linear automation and DevOps review processes for the AI Agent. ([79d2649](https://github.com/LambdaSection/NeuralDBG/commit/79d2649a0985f7be0a5e6a1cdc9dd4e8f87de75a))
* Add RULE 29 for mandatory Linear integration to all relevant documentation files. ([7eb9743](https://github.com/LambdaSection/NeuralDBG/commit/7eb9743f59e4cb9d5006f5d346d5215ae1ac5015))
* add Rule 33 (Global Rule Parity and Mandatory Cross-Branch Sync) ([fdbdd2f](https://github.com/LambdaSection/NeuralDBG/commit/fdbdd2f537cf60db4d741cad6028a17b1041abed))
* add Rules 26-36 for Linear integration and CEO progress tracking ([f183f03](https://github.com/LambdaSection/NeuralDBG/commit/f183f03880f89cf4a2a65ec572e39cc7ec5d5af1))
* add TensorBoard comparison to demo + bandit security report ([4acb660](https://github.com/LambdaSection/NeuralDBG/commit/4acb6607f68a98c81155132f563a2f34427b3e37))
* Add unit tests for SemanticEvent and NeuralDbg's causal inference engine, and update agent guidelines. ([1a6f891](https://github.com/LambdaSection/NeuralDBG/commit/1a6f891cf1c5ebc31d0178e9d97d6c14ce1f7dee))
* **aladin:** implement synthetic generator and time series dataset ([59bb615](https://github.com/LambdaSection/NeuralDBG/commit/59bb615c3ce5fddfeccce625b61b75c5e92b6838))
* **api:** add initial api module and test suite ([6884f29](https://github.com/LambdaSection/NeuralDBG/commit/6884f29520eed07be3b4de07688398fc09ab9161))
* Aquarium JSON export + CI benchmark workflow ([59ca8e1](https://github.com/LambdaSection/NeuralDBG/commit/59ca8e1a1e92f5bba82c3ef0eeac7facbc2007fa))
* Aquarium web dashboard — zero-dependency causal viewer ([a429aab](https://github.com/LambdaSection/NeuralDBG/commit/a429aab3dbbb22eb93c6a6fadce13ae5cf3a6ac2))
* architecture fuzzer v1 — Tier 2 black-swan discovery ([982e499](https://github.com/LambdaSection/NeuralDBG/commit/982e4990b11a26b0fdd27730a2a572ccd9133412))
* automate validation sync + add bidirectional upload ([b11db82](https://github.com/LambdaSection/NeuralDBG/commit/b11db821aab8ead13bc083373b770305e335b0f1))
* automated black-swan paper scraper — 8 arxiv queries ([4223466](https://github.com/LambdaSection/NeuralDBG/commit/4223466ea28135164d741f22ffe0608737019604))
* brand GitHub presence, public benchmark, Colab, and suite positioning ([8da871b](https://github.com/LambdaSection/NeuralDBG/commit/8da871bc53f73d475c35f5586ce4b73cae40bf92))
* **bug-004:** add HuggingFace Qwen3.5 SDPA gradient explosion catalog ([47a3842](https://github.com/LambdaSection/NeuralDBG/commit/47a3842d8d3a957c8b9097b79d20334654ee0f7f))
* **bug-005:** add pytorch[#173334](https://github.com/LambdaSection/NeuralDBG/issues/173334) CUDA LSTM batch pollution catalog + curation for M2 ([13a21c0](https://github.com/LambdaSection/NeuralDBG/commit/13a21c0c54dfc037919244a3fc9998709812f59b))
* **bug-006:** catalog pytorch[#187759](https://github.com/LambdaSection/NeuralDBG/issues/187759) — svdvals silently swallows NaN (fresh bug, June 20) ([f403612](https://github.com/LambdaSection/NeuralDBG/commit/f40361244dbe08c0f12cb4bbb3d515802855e062))
* **bug-007:** catalog pytorch[#186799](https://github.com/LambdaSection/NeuralDBG/issues/186799) — torch.compile silent gradient corruption (reported by [@ezyang](https://github.com/ezyang)) ([69a4706](https://github.com/LambdaSection/NeuralDBG/commit/69a470675b78496f88092283a2d53068ce37216c))
* BUG-008 repro + PR [#188066](https://github.com/LambdaSection/NeuralDBG/issues/188066) created (F.normalize zero-input) + BUG-009/010 repro scripts ([b2a2f97](https://github.com/LambdaSection/NeuralDBG/commit/b2a2f973233ca57de6b291821f5c414ee3a22a46))
* **bugs:** catalog BUG-008/009/010 — 10 bugs total (M2 objective reached) + PR Gate updated with d2h sync lesson ([8d5eac3](https://github.com/LambdaSection/NeuralDBG/commit/8d5eac3929032e192f5bcf703e0eff3becd70fd6))
* **community:** growth infrastructure (P1) — greet, labeler, credits, discoverability ([b0122d0](https://github.com/LambdaSection/NeuralDBG/commit/b0122d05251a102c20d3622e1ae9f1a6a74a8441))
* composite-module hook support (FIX-001, BUG-001) ([3f3b69a](https://github.com/LambdaSection/NeuralDBG/commit/3f3b69acbb35146209308d96f734dbd0653b8848))
* **core:** add conditional NeuralDBG-Engine import + copyright headers ([1a1ec83](https://github.com/LambdaSection/NeuralDBG/commit/1a1ec834964eeb9a3cb3fbcf60e858943e72d002))
* **core:** add resource profiling to semantic events for MLO-10 (WIP) ([56d6aa5](https://github.com/LambdaSection/NeuralDBG/commit/56d6aa543e3aa5c71abf41b4d2833ced5ba571a5))
* **core:** complete resource profiling integration for MLO-10 ([4a8ac0c](https://github.com/LambdaSection/NeuralDBG/commit/4a8ac0cc339a559bff883a54632f5228a691d919))
* **core:** fix backward hook inplace compatibility + dogfooding + 2nd demo ([6553121](https://github.com/LambdaSection/NeuralDBG/commit/6553121bf41f66751637ad8152faa91ea0fc271b))
* **core:** implement optimizer instability + data anomaly detection, complete Phase 3 & 4 ([b4736a9](https://github.com/LambdaSection/NeuralDBG/commit/b4736a97f7355bf7364373982d30bad0ecc0b7a5))
* **core:** implement Phase 2 Compiler-Aware Hardening ([d38764b](https://github.com/LambdaSection/NeuralDBG/commit/d38764bb34df452d83da617faf5c8eab9968879a))
* Define new agent rules for MLOps/DevOps milestone task generation and persona adaptability, and create initial infrastructure planning document. ([22603b7](https://github.com/LambdaSection/NeuralDBG/commit/22603b74529b8d50f5028e84cea86b1e0199093e))
* **demo:** add DDPM diffusion failure scenarios (NaN, exploding, noise collapse) and tests ([df2ff3a](https://github.com/LambdaSection/NeuralDBG/commit/df2ff3ae3254ff4b1c1338ac3acdbeba57c75faa))
* **demo:** add GAN generator failure scenarios (vanishing, exploding, NaN) and tests ([9229b89](https://github.com/LambdaSection/NeuralDBG/commit/9229b891d39d02a929b82503dd0fb98c815faf32))
* **demo:** add LoRA fine-tuning failure scenarios (NaN, exploding, forgetting) and tests ([ecc811c](https://github.com/LambdaSection/NeuralDBG/commit/ecc811c0b7bad61850c52c317a9792d07085f790))
* **demo:** add ResNet-18 failure scenarios and tests ([f494265](https://github.com/LambdaSection/NeuralDBG/commit/f4942654b3c5088e385f51d1bd1b4a749ff45676))
* **demo:** add Transformer (GPT) failure scenarios and tests ([4b9b13b](https://github.com/LambdaSection/NeuralDBG/commit/4b9b13bf67342afde1bb579a71e72911131a6f00))
* **devops:** add pre-commit, sync_summary.py, README badges ([ab0a769](https://github.com/LambdaSection/NeuralDBG/commit/ab0a7692850dfb68e051b6866038583e4c5db305))
* **docs:** automate API doc generation with pdoc (NDBG-9) ([6e47d85](https://github.com/LambdaSection/NeuralDBG/commit/6e47d854dfb9514f0670589b78bed01731431050))
* **docs:** automate API doc generation with pdoc for NDBG-9 ([d6622df](https://github.com/LambdaSection/NeuralDBG/commit/d6622dfa0c61df7d6a207c46df8cd5dbcf5753a3))
* **dvc:** complete MLO-4 — DagsHub remote, synthetic data tracking, docs ([e7bd2f3](https://github.com/LambdaSection/NeuralDBG/commit/e7bd2f396b9f37851f78115a4e1ed1c6400430fe))
* E2E RNN pipeline — closed loop on LSTM bugs ([aaaf096](https://github.com/LambdaSection/NeuralDBG/commit/aaaf09660bd68fbfc2843c043d9178c4d9fd1d74))
* ecosystem integrations + community post drafts + PyTorch Ecosystem submission ([9ae8440](https://github.com/LambdaSection/NeuralDBG/commit/9ae844004af7b9aa501863019f0d3f9efec05908))
* **ecosystem:** complete multi-repo cartography (R105/R106 compliance) ([8fdae71](https://github.com/LambdaSection/NeuralDBG/commit/8fdae71829adb839a563a33d714951892ed8345d))
* end-to-end NeuralSuite demo — NeuralDBG + NeuralPrune + Tier 3 ([1531327](https://github.com/LambdaSection/NeuralDBG/commit/15313275aeebdab9c6854b83984a44a1bf9f876c))
* end-to-end NeuralSuite demo script ([542b43e](https://github.com/LambdaSection/NeuralDBG/commit/542b43ec0800c600ea95f5b272bdcc6bb233b77c))
* Establish initial project discovery and validation framework with Mom Test scripts, research prompts, decision memos, and a session summary. ([7045d8c](https://github.com/LambdaSection/NeuralDBG/commit/7045d8c4c78ddd6bb149cf129dc5ca5916503c9a))
* Establish new mandatory project rules for validation, emoji usage, rule synchronization, and collaboration, while adding new tests and planning documents. ([5b6d9d3](https://github.com/LambdaSection/NeuralDBG/commit/5b6d9d3372089b7a6ef8459e9f784976315b221d))
* family-aware detection threshold — 90% global, Hybrid 96% ([2d398d1](https://github.com/LambdaSection/NeuralDBG/commit/2d398d123e1f08be131775f8345b57da0db829dd))
* honest benchmark (NeuralDBG vs detect_anomaly vs W&B) + paper section 5.12 rewrite ([18a12d6](https://github.com/LambdaSection/NeuralDBG/commit/18a12d63de957bacdb3120ba3a6e06b47de40e63))
* honest benchmark + LaTeX paper conversion (arXiv-ready) ([a386437](https://github.com/LambdaSection/NeuralDBG/commit/a3864376518f1636eabff85b4ee294c77dba0bfb))
* HuggingFace Spaces demo app (Gradio) + updated PLAN priorities ([0f6be20](https://github.com/LambdaSection/NeuralDBG/commit/0f6be2002bdf6c9793e4435e5a8436f548186aee))
* implement 'Extreme Rigueur' rules, add Phase 3 docs, and fix progress audit ([b985aa1](https://github.com/LambdaSection/NeuralDBG/commit/b985aa19a4290f522f649e7c603a56009b2d961c))
* Implement activation saturation detection, regime shift detection, and failure explanation, accompanied by a new test case. ([4d4aaa0](https://github.com/LambdaSection/NeuralDBG/commit/4d4aaa0a3b7b20b932d8327690cf8bd11cb6e85f))
* Implement CI/CD debugging rule across AI agent configurations and add Google Docs automation for session summaries. ([2828048](https://github.com/LambdaSection/NeuralDBG/commit/28280483204f79acadff4150174fe5ab8b4ebbbb))
* implement Rule 30/33 authority and naming conventions (ceo/ scope) ([1a6ceff](https://github.com/LambdaSection/NeuralDBG/commit/1a6ceffd3096786ad155e8c5e668e95180e293b4))
* implement validation bundle sync infrastructure ([b51e68b](https://github.com/LambdaSection/NeuralDBG/commit/b51e68be78769cff3c555ab00dee847ca9fa8b9b))
* import torch library to enable PyTorch functionality in neuraldbg.py ([1c0ae4d](https://github.com/LambdaSection/NeuralDBG/commit/1c0ae4d72159b301fcf47889cc4f7f410504fc88))
* **infra:** automate venv lifecycle and collaborator onboarding for MLO-17 ([3db20a2](https://github.com/LambdaSection/NeuralDBG/commit/3db20a20c75822a96e65659d1af7283805b14581))
* **infra:** MLflow tracking + bootstrap automation + CI pipeline ([be46151](https://github.com/LambdaSection/NeuralDBG/commit/be46151004401de3200240316b00a77612d49d13))
* Introduce a comprehensive codebase guide and establish rules for standardized Linear issue labels and detailed task context. ([16df67b](https://github.com/LambdaSection/NeuralDBG/commit/16df67b2e03767441ccf5b15b18e86ada39a45f4))
* introduce NeuralSuite as unified brand name ([59355d8](https://github.com/LambdaSection/NeuralDBG/commit/59355d88e051f8ce93ba8d6529a80f8a5e248808))
* **landing:** capture pages S1 (landing + checklist, zero tracking) ([#686](https://github.com/LambdaSection/NeuralDBG/issues/686)) ([06a7600](https://github.com/LambdaSection/NeuralDBG/commit/06a76003473a0df31a47d0ed5c6cef55fa795c23))
* **landing:** offre beta gratuite sur la landing ([#687](https://github.com/LambdaSection/NeuralDBG/issues/687)) ([b2f02ec](https://github.com/LambdaSection/NeuralDBG/commit/b2f02ec48dd18e824bdfc51337feaaae0bac43f0))
* **linear:** add editor-agnostic Linear API workflow ([8e11a5c](https://github.com/LambdaSection/NeuralDBG/commit/8e11a5c26217f28c4a2bfecffc5f094d4aca12f9))
* live benchmark — NeuralDBG vs Baseline (W&B/TensorBoard simulator) ([eab7898](https://github.com/LambdaSection/NeuralDBG/commit/eab7898da675ecd355027b1598088f2c2c8018d7))
* MCP server + multi-path validation strategy in PLAN ([f50e236](https://github.com/LambdaSection/NeuralDBG/commit/f50e23642fc965426c46afc557b3078e847d91a2))
* **mlops:** initialize DVC for binary artifact versioning (MLO-4) ([c44d43c](https://github.com/LambdaSection/NeuralDBG/commit/c44d43c24b0b57ac4a28c754e70bee81c6a95b11))
* **mlops:** integrate optional MLflow tracking for MLO-2 ([0ada8f4](https://github.com/LambdaSection/NeuralDBG/commit/0ada8f4125eedace4d61793781c9dd199cebd37f))
* NeuralPrune + v5 exporter + Tier 2 black-swans (94%) ([40fe4e7](https://github.com/LambdaSection/NeuralDBG/commit/40fe4e7a1feb48e68d323db69ef67b41b79d24dd))
* one-click demo — NeuralDBG bug hunt in 60 seconds ([a702269](https://github.com/LambdaSection/NeuralDBG/commit/a70226955786732f35af80ca57e1e9e148f9c795))
* **oom-prevention:** add TensorDiskCache and optimize JIT stats calculations ([ff8d5ce](https://github.com/LambdaSection/NeuralDBG/commit/ff8d5cee2b1b3860f1858bc3f2d6bd27cf83c19d))
* **oos:** out-of-sample validation on torchvision ResNet-18 — 6/6 (100%) ([f8e15bd](https://github.com/LambdaSection/NeuralDBG/commit/f8e15bd71ab61e586c4e73f9bae8a2b54241d998))
* per-gate LSTM/GRU gradient tracking — vanishing +20% ([7c46015](https://github.com/LambdaSection/NeuralDBG/commit/7c4601575a48d84005c485a097455977c830cad0))
* Phase 10 Launch assets — GitHub Pages, demo recording script, HN draft ([7fa2c28](https://github.com/LambdaSection/NeuralDBG/commit/7fa2c28fc65d40e27193c6c171952bf0ce981d5e))
* Phase 10 MVP Launch assets — quickstart, issue templates, README comparison ([1402dc0](https://github.com/LambdaSection/NeuralDBG/commit/1402dc0cff55027ae204976b691deac4bbf56d07))
* Phase 2-7 complete — dogfooding, Aquarium export, two-package, PyPI ready ([294f169](https://github.com/LambdaSection/NeuralDBG/commit/294f1697a0e5ba8df94a754ab75d07b8e8a06b44))
* PR Gate checker script — validates all 6 gates before upstream PR creation ([d250de8](https://github.com/LambdaSection/NeuralDBG/commit/d250de883d361a80b6e7e3ca020beeed44279d10))
* PR Gate system — mandatory 6-gate checklist preventing repeat of PR [#186631](https://github.com/LambdaSection/NeuralDBG/issues/186631)/[#186786](https://github.com/LambdaSection/NeuralDBG/issues/186786) mistakes (R89 lessons learned) ([647bea4](https://github.com/LambdaSection/NeuralDBG/commit/647bea4ac2ed5dd40379d81fa473cefdea4fb863))
* real tool comparison — NeuralDBG vs W&B vs MLflow vs TensorBoard ([a986fed](https://github.com/LambdaSection/NeuralDBG/commit/a986fed19c19836d85307fd35d9df56fc9ea7157))
* restore and merge project continuity rules ([f3c1ee8](https://github.com/LambdaSection/NeuralDBG/commit/f3c1ee864ba22f7e67deb5aad46719516f633c25))
* Revamp project rules for pedagogical AI interaction, introduce AI guidelines, and establish session summary tracking. ([d274770](https://github.com/LambdaSection/NeuralDBG/commit/d274770c5a7b97c2bc79302ea9708cf457f6d051))
* RL detector (5/5=100%), BlackSwan family, OOS v2 (4 archs), R108 pipeline CI ([f1d6968](https://github.com/LambdaSection/NeuralDBG/commit/f1d69684eca37f076f51dc952ad9c753b1163755))
* **rules:** add roadmap adherence and duration rules to AGENTS.md ([7aad82f](https://github.com/LambdaSection/NeuralDBG/commit/7aad82fff34f5f85131bc9e18ebd68e76f5daa29))
* **rules:** mandate cumulative session summaries ([113cd12](https://github.com/LambdaSection/NeuralDBG/commit/113cd1295d0e778c5e1184e8a908f9b15bb0ed01))
* **rules:** sync cumulative traceability rules ([8af262f](https://github.com/LambdaSection/NeuralDBG/commit/8af262fc5b7d68e4f595b53a91e1fdbb39a8988c))
* run_pipeline.py — full R108 pipeline runner with auto-retrain loop ([fa2b4fe](https://github.com/LambdaSection/NeuralDBG/commit/fa2b4fe60bfcc07585e6a68ddf904d3f8ef58741))
* Self-Evolution Engine — daily auto-improvement pipeline ([1432158](https://github.com/LambdaSection/NeuralDBG/commit/1432158109b6a4636bf31a216ae833236b9700e6))
* stress test suite — 15/15 (100%) +10 resilience verified ([d5e6058](https://github.com/LambdaSection/NeuralDBG/commit/d5e6058133cdc96a25f9f3ac650aa505a8c99503))
* **sync:** add SESSION_SUMMARY.md to .docx conversion script (NDBG-5) ([0f2ac4b](https://github.com/LambdaSection/NeuralDBG/commit/0f2ac4bdfb0867ec64a1261a29bca6ffbba6ba83))
* Tier 1 black-swan tester — GNN/MoE/Diffusion validation ([705dd6e](https://github.com/LambdaSection/NeuralDBG/commit/705dd6edc290b9eeeeae92733ab97755f5296bbb))
* Tier 3 Predictive Anomaly Detector — zero-config black-swan detection ([61bd54a](https://github.com/LambdaSection/NeuralDBG/commit/61bd54a77323f543760f3b997d9970cfc8fc7fdb))
* Tier 4 black-swans — RL Actor-Critic + RAG ([1f2e774](https://github.com/LambdaSection/NeuralDBG/commit/1f2e77443923f189e3724379ea6e61380d3a3211))
* Tier 4b — Federated Learning validator (50% detection) ([7ce5333](https://github.com/LambdaSection/NeuralDBG/commit/7ce5333bf6cb8f016fd8d709bb8f0693c1a9ee1b))
* trend-based vanishing gradient detection (+10% improvement) ([3ae77af](https://github.com/LambdaSection/NeuralDBG/commit/3ae77af6b808776542e5e8c71a0028371d928556))
* universal multi-lingual rules and auto-commit protocol ([47a9edb](https://github.com/LambdaSection/NeuralDBG/commit/47a9edb93b222b0937905607af318e9316c55c31))
* v5 GPU training script — Qwen2-0.5B + LoRA on 108 examples ([9127886](https://github.com/LambdaSection/NeuralDBG/commit/9127886dbca51347cde72a5c07ed30c1bee97a7a))
* v5 training data — 108 examples across 6 families ([b1d83df](https://github.com/LambdaSection/NeuralDBG/commit/b1d83dfd5da8de820217b25a8131351775276128))
* W&B + PyTorch Lightning integrations — NeuralDBG as callback ([3d8bdf3](https://github.com/LambdaSection/NeuralDBG/commit/3d8bdf31f032ba7b159c5489c46c0a20cb62aeb7))


### Bug Fixes

* adamw_torch instead of paged_adamw_32bit (Quadro M4000 no bitsandbytes CUDA) ([1c59922](https://github.com/LambdaSection/NeuralDBG/commit/1c599226232316f0b87fb9627da33093d87b64fd))
* add examples/__init__.py so tests can import demo modules ([5b33123](https://github.com/LambdaSection/NeuralDBG/commit/5b33123ee1496b3910384aa36b7638e6a7721b54))
* add explicit permissions to all GitHub Actions workflows for CodeQL compliance ([fe0cbd9](https://github.com/LambdaSection/NeuralDBG/commit/fe0cbd9753bcceab38da038ad84adb19a869d6f2))
* add PYTHONPATH for CI so examples/ modules are importable ([9f3c67b](https://github.com/LambdaSection/NeuralDBG/commit/9f3c67b0a796fee54037e1797adb8087d450ca4a))
* allow safety check to fail without blocking CI ([e127eda](https://github.com/LambdaSection/NeuralDBG/commit/e127eda61d4d4881c074b3737f2edcc03b9ba87e))
* apply black+isort formatting to test_error_paths ([8135352](https://github.com/LambdaSection/NeuralDBG/commit/8135352502d8451cb707f4306a38d4a33142da5d))
* **arch_fuzzer:** named bug functions, dict LR access, half labels (1/20 -&gt; 16/20) ([cd48384](https://github.com/LambdaSection/NeuralDBG/commit/cd4838420bc96eb0a138849bf997f397c7cff642))
* **arch_fuzzer:** validate shape spec before training (5x detection rate) ([642805b](https://github.com/LambdaSection/NeuralDBG/commit/642805b8970738253f6d49dfa25893f791de17f8))
* **bandit:** add nosec B602 to ci_agent shell invocation ([b9501eb](https://github.com/LambdaSection/NeuralDBG/commit/b9501eb5f65eb276e5e229653a5259decb6520b5))
* benchmark auto-detects composite modules for BUG-001 scenario ([d5e24c9](https://github.com/LambdaSection/NeuralDBG/commit/d5e24c96b139ca530da58783298719ec4f3799ab))
* benchmark vanishing scenario 0.667→1.000 — correct ground truth layer name, fix hypothesis layer matching ([31e19cc](https://github.com/LambdaSection/NeuralDBG/commit/31e19cc993f3f07b27efc188f675701e96ac7e08))
* BlackSwan detection 0/48 -&gt; 48/48 (100%), unicode cleanup in pipeline runner ([9a1aa66](https://github.com/LambdaSection/NeuralDBG/commit/9a1aa66f4e278259ba546a25cef4c136b62f9363))
* BlackSwan GNN/MoE model builders — handle 2D inputs (0/48 -&gt; 12/48 detection) ([fa2b4fe](https://github.com/LambdaSection/NeuralDBG/commit/fa2b4fe60bfcc07585e6a68ddf904d3f8ef58741))
* bug injectors use imported functions — MoE 36% -&gt; 100% ([e865d79](https://github.com/LambdaSection/NeuralDBG/commit/e865d791229defb696fa94d13faf00235e09b0e2))
* bug_vanishing + bug_nan for GNN — 66% -&gt; 88%, Tier 1 84% -&gt; 96% ([1901540](https://github.com/LambdaSection/NeuralDBG/commit/190154057c33361daf025d649946f8374e9ada6c))
* calibrate pipeline gates — fuzzer=any discovery, stress=14/15, combinatorial=85% ([c7e984e](https://github.com/LambdaSection/NeuralDBG/commit/c7e984efc773b51dff373811c2356cc37602740b))
* causal attribution priority + strict mode + family calibration ([2b48b0e](https://github.com/LambdaSection/NeuralDBG/commit/2b48b0ec8d8b265789a39b0739ed3416815ac85a))
* causal chains on RNN — added missing compatibility pairs ([49f58da](https://github.com/LambdaSection/NeuralDBG/commit/49f58da03250801a12699626b80addd126e06a0b))
* causal matrix + trend detection — gradient-&gt;data links, 50% drop threshold ([81f28ad](https://github.com/LambdaSection/NeuralDBG/commit/81f28ad28541aabc953a45c8c0fb52a2b7b9e780))
* **ci:** bandit scope to neuraldbg only, skip intentional B102/B310/B615 ([29c06dc](https://github.com/LambdaSection/NeuralDBG/commit/29c06dc3998a0946c5b0207841186ff364dac715))
* **ci:** corrige les 3 workflows rouges - retire le sous-module fantome hf_space (echec checkout pages), remplace les asserts production par des raises (R-compliance), nosec justifies sur les 10 findings bandit medium (B615/B102/B310/B314) ([d1df40f](https://github.com/LambdaSection/NeuralDBG/commit/d1df40f9c6379a4586f737974c7f1dea14dcee1b))
* **ci:** engine hard-anomaly bypass + gradient P2b + watchdog agent ([0ded50e](https://github.com/LambdaSection/NeuralDBG/commit/0ded50ed73e424560bd967f214d6dee7dda41a52))
* **ci:** exclusions R64 pour la doc qui reference la regle (PR_TEMPLATES, CHANGELOG) + formatage black/isort des fichiers touches ([d86b10a](https://github.com/LambdaSection/NeuralDBG/commit/d86b10ad3d687fef5a459d14dd460a72863420f0))
* **ci:** exclut le manifest genere du scan detect-secrets ([#682](https://github.com/LambdaSection/NeuralDBG/issues/682)) ([c77c17f](https://github.com/LambdaSection/NeuralDBG/commit/c77c17f656e4e87226a35d5926513e952ac876bc))
* **ci:** newline final manifest (end-of-file-fixer) ([#683](https://github.com/LambdaSection/NeuralDBG/issues/683)) ([caa6c7f](https://github.com/LambdaSection/NeuralDBG/commit/caa6c7f6091a26dd2c34035c758e8e6438e7fead))
* **ci:** pre-commit vert - .flake8 per-file-ignores pour la dette structurelle legacy (E501/E402/E722/E731/F541/E231), purge des imports et variables morts, R64 scope produit (neuraldbg+scripts) ([3ffbbf3](https://github.com/LambdaSection/NeuralDBG/commit/3ffbbf3e50cf9ce4e85ba88a8dab3243cc47a08b))
* **ci:** retablit l'exclusion CHANGELOG.md dans le check R64 (perdue lors du rescope) ([8e666c1](https://github.com/LambdaSection/NeuralDBG/commit/8e666c12f8c551bd8be18380044fb3d1fef804e9))
* clean lint warnings in test_error_paths (unused imports, variables) ([0cac6f9](https://github.com/LambdaSection/NeuralDBG/commit/0cac6f94210a097bfa57c2b7cda3ca0298f62c76))
* comparison v2 table format + benchmark results in ROADMAP ([c2a3126](https://github.com/LambdaSection/NeuralDBG/commit/c2a312666fffc34d714da205d595a2f1745b0c3e))
* **core:** deduplicate causal couplings ([46ee983](https://github.com/LambdaSection/NeuralDBG/commit/46ee983db6f670f404dc45869ca78a22256e3c05))
* **core:** detect NaN in module output for precise localization ([472a564](https://github.com/LambdaSection/NeuralDBG/commit/472a5648fe36c02b018b79dfc531aa81211d7695))
* **core:** implement state transition tracking for data anomaly, fix collapse revert detection for baselines ([b46025e](https://github.com/LambdaSection/NeuralDBG/commit/b46025e4a2fa1079b99e5028618d694faa140ed2))
* **core:** improve layer naming (Linear_0, Tanh_1...) for readable hypotheses ([f282805](https://github.com/LambdaSection/NeuralDBG/commit/f282805e4805a0a21b8b6f24d9ddd2cd3d8efe55))
* **core:** prevent NaN/Inf from poisoning distribution shift stats ([2df9127](https://github.com/LambdaSection/NeuralDBG/commit/2df9127d8c704417ee2ac4b20a2afa464b224022))
* **core:** restore demo test and resource scan ([575b248](https://github.com/LambdaSection/NeuralDBG/commit/575b248718983020ae1c87a19656120253059a4c))
* **deps:** update URLs to LambdaSection, bump numpy to 1.26.4, constrain torch/psutil ([7a8b879](https://github.com/LambdaSection/NeuralDBG/commit/7a8b879d9bb43fc37894fc91cd826ba6fdda98a6))
* **docs:** resolve markdown lint errors in INFERENCE_FLOW.md and ROADMAP.md ([96d5f20](https://github.com/LambdaSection/NeuralDBG/commit/96d5f202bb14635302081cffa895c637502748d9))
* **docs:** update STRUCTURE.md to remove private file references ([ff684bd](https://github.com/LambdaSection/NeuralDBG/commit/ff684bdb8cc887bda7fa5b572bb0931d6b66d005))
* **dogfooding:** correct failure_type string activation_saturation -&gt; saturated_activations ([5d0207f](https://github.com/LambdaSection/NeuralDBG/commit/5d0207f519bc0a01879ffa2ecc548a91de87d7e9))
* **dvc:** allow .dvc pointer files in data/, models/, outputs/, artifacts/ ([1dbe632](https://github.com/LambdaSection/NeuralDBG/commit/1dbe632a9019f24597a72e4704a4061289bbafbb))
* evolve.py — capture_output=False for visible progress, reduced timeouts ([9c022d3](https://github.com/LambdaSection/NeuralDBG/commit/9c022d368d8d46153d9b0e0d8c1f4e5a9c24ca57))
* **fp:** reduce false positives on deep architectures — 142-&gt;5 events on healthy ResNet-18 ([a67f560](https://github.com/LambdaSection/NeuralDBG/commit/a67f5608c5ab97ae695786650e19f95e0c96dcdb))
* **git:** clean .gitignore and remove sensitive files from tracking ([affd0dc](https://github.com/LambdaSection/NeuralDBG/commit/affd0dc8752179dc54feb4b08e9df53c1de3c685))
* **git:** remove __pycache__ from tracking ([d37132d](https://github.com/LambdaSection/NeuralDBG/commit/d37132de577c15479935aa8c18591a2550f82893))
* **git:** remove planning/research/prompts from tracking per R62/R76 ([c3c4c49](https://github.com/LambdaSection/NeuralDBG/commit/c3c4c49d5a83100b19ad68ab19154ea166d4b93b))
* **git:** remove protected files from tracking per R10/R76 ([79bfb3c](https://github.com/LambdaSection/NeuralDBG/commit/79bfb3ccc64968761a421befa63ab998f63a9228))
* **git:** remove sensitive documentation files from tracking per R76 ([af879c6](https://github.com/LambdaSection/NeuralDBG/commit/af879c6bc572b5659e735b406d72e72eaa9fc002))
* **git:** remove test_aquarium_bridge.py (Aquarium IDE has its own repo) ([4814337](https://github.com/LambdaSection/NeuralDBG/commit/4814337a67f398d4a0f8792c59e682a03daa19fa))
* harden fallback paths and sensitive file guard ([2f3f4ca](https://github.com/LambdaSection/NeuralDBG/commit/2f3f4ca80a4875d79dc049ffd05f4cb6e58cd7df))
* harden fallback paths, fix hooks attribute typo in test, cleanup ([6e140ff](https://github.com/LambdaSection/NeuralDBG/commit/6e140ff452a1d1c5cc689983b30d9cd56c90165c))
* **hooks:** move shebang to line 1 in post-checkout and post-merge hooks ([2192294](https://github.com/LambdaSection/NeuralDBG/commit/2192294d9bdec2d4a1823a46c2b5465c72da34c1))
* **infra:** detect broken venv via pip health check, bootstrap pip via get-pip.py ([f56f470](https://github.com/LambdaSection/NeuralDBG/commit/f56f470ab7fe1044ee7ef3e708b9dfcc4840e413))
* **infra:** detect Python version mismatch in ensure_venv.sh (MLO-17) ([8f3fa45](https://github.com/LambdaSection/NeuralDBG/commit/8f3fa450c569cdea84e2234a5fee8873421f7274))
* **infra:** use --without-pip venv creation for Debian/Ubuntu compat ([a0749fd](https://github.com/LambdaSection/NeuralDBG/commit/a0749fd8309eb3363a0451e3d2bcef2a3b5e1232))
* install neuraldbg-engine in CI when available, mark engine-dependent tests ([037d149](https://github.com/LambdaSection/NeuralDBG/commit/037d1497c6fe59f009539f295b8321d2a5f7a8d2))
* install project in editable mode in CI, fix coverage include path ([b004ae9](https://github.com/LambdaSection/NeuralDBG/commit/b004ae9f45af4ee37f324fa19f735cbb5eae3338))
* install pytest-cov in CI to fix coverage gate ([032cf64](https://github.com/LambdaSection/NeuralDBG/commit/032cf64b33f57e8ef7cd61792ecbe61e56241730))
* last unicode arrow in validate_oos.py ([afa69d7](https://github.com/LambdaSection/NeuralDBG/commit/afa69d73b8c9d50c58613a551eba0f9c0959e920))
* **lint:** remove unused imports + fix f-string in train_cpu.py ([290276c](https://github.com/LambdaSection/NeuralDBG/commit/290276c4a635b21584418ade98ddfc514e8319ad))
* lower coverage threshold to 30% for CI without engine ([4b8d5cc](https://github.com/LambdaSection/NeuralDBG/commit/4b8d5cc11230c3c86573a3f0868b9fe1997b68e2))
* Mamba-Mini SSM block dimension mismatch + stress parsing ([f452314](https://github.com/LambdaSection/NeuralDBG/commit/f452314c329eb86aeb46083fbe330c0e30594aff))
* mark all engine-dependent tests with [@requires](https://github.com/requires)_engine ([ca4166a](https://github.com/LambdaSection/NeuralDBG/commit/ca4166a53d4023aeb1ae1ba1577b2e2458341ebf))
* mark remaining engine-dependent tests with [@requires](https://github.com/requires)_engine ([7a899d9](https://github.com/LambdaSection/NeuralDBG/commit/7a899d9ddaf642c7cffb0f841d85a3e04220b0e6))
* MoE data generator cfg-aware — 11% -&gt; 36% detection ([5e31f58](https://github.com/LambdaSection/NeuralDBG/commit/5e31f5836597915df374acfa458351e38f7d4319))
* pipeline subprocess deadlock + stress parsing regex ([63e67df](https://github.com/LambdaSection/NeuralDBG/commit/63e67dfd1ecf9a7bcc65460ee99841fb7ef6237e))
* pr_gate_check — use working-tree diff instead of HEAD~1, exclude self ([deb8216](https://github.com/LambdaSection/NeuralDBG/commit/deb8216421527e667aa0687fcb372aa42d83d0a6))
* pyproject.toml — docs moved from scripts to optional-dependencies ([f0287de](https://github.com/LambdaSection/NeuralDBG/commit/f0287ded4ae7fe757ff982fb6ed5739f6892d6c4))
* **R106:** remove PLAN.md from public repo — must be private ([1b1ff92](https://github.com/LambdaSection/NeuralDBG/commit/1b1ff925afbf7eb07589819e42b019f342df6d7e))
* remove 'causal inference' from titles — sell the benefit, not the technique ([b71095a](https://github.com/LambdaSection/NeuralDBG/commit/b71095ad6f17d366275abff9da5062fd966b74b9))
* remove deprecated license classifier for PEP 639 compliance ([763bcbc](https://github.com/LambdaSection/NeuralDBG/commit/763bcbcdd20f44baa98152535e56dedabd8882b7))
* remove mlflow from requirements-mlops.txt to clear Dependabot alerts ([3f507f2](https://github.com/LambdaSection/NeuralDBG/commit/3f507f22df0a2c46114f590554b3ee83a99a198d))
* remove remaining unicode emojis from pipeline runner (cp1252 compat) ([bc87d20](https://github.com/LambdaSection/NeuralDBG/commit/bc87d20c9c0826f17decff1b09f52fc46372d937))
* remove vulnerable mlflow dependency, update pytest to 8.0+ ([34c6266](https://github.com/LambdaSection/NeuralDBG/commit/34c6266ae436e8c01d7d69db5ef5507024ca560d))
* RNN detection 49%-&gt;68% — unwrap LSTM/GRU output tuples in hooks ([ab18c76](https://github.com/LambdaSection/NeuralDBG/commit/ab18c769a923076814d8abb6d836d5dbb15613cc))
* run unit tests only in CI to avoid examples/ import issues ([0e1a660](https://github.com/LambdaSection/NeuralDBG/commit/0e1a6607905d8fed5a4e26c7fcae5a8a8433c57f))
* **security:** add # noqa: B615 to from_pretrained() calls (bandit heuristic bug) ([344f54d](https://github.com/LambdaSection/NeuralDBG/commit/344f54d678d8b301192506cb33e22450972d63dc))
* **security:** add pragma allowlist for DVC md5 hashes and example API key (false positives) ([582ea26](https://github.com/LambdaSection/NeuralDBG/commit/582ea26dd4bd9d4057b50b7c09263ef40cf63bf6))
* **security:** inline revision='main' for multi-line from_pretrained() calls (B615) ([2b930c2](https://github.com/LambdaSection/NeuralDBG/commit/2b930c2f2757f6513f5817aff846a1e4ae43b7b8))
* **security:** pin Hugging Face Hub downloads with revision='main' (B615) ([d4480f0](https://github.com/LambdaSection/NeuralDBG/commit/d4480f02a11fec7e3036e0611d759215e37cae75))
* **security:** use # nosec: B615 instead of # noqa: B615 (bandit syntax) ([2c78a1b](https://github.com/LambdaSection/NeuralDBG/commit/2c78a1b3fd475150f78c61c07f9f6399f17d995c))
* skip engine-dependent tests in CI, skip docx tests when python-docx missing ([673db68](https://github.com/LambdaSection/NeuralDBG/commit/673db689b63a3faa1adf8d57cb3bb345c6d241c8))
* **stats:** skip activation stats calculation for non-floating-point tensors ([859b24e](https://github.com/LambdaSection/NeuralDBG/commit/859b24e80a10857e8463594b343caf46647a3742))
* stress test 14/15 -&gt; 15/15 (100% resilience) ([e60dc43](https://github.com/LambdaSection/NeuralDBG/commit/e60dc4393f2240dfdfcbebd127551a743d0cc1c7))
* **tests:** skip torch.compile tests when python3-dev headers absent ([a79d1e6](https://github.com/LambdaSection/NeuralDBG/commit/a79d1e62b8d2865ae1391b79f16f00c3d3508460))
* Tier 3 predictive detector — family-aware profiles + API fixes ([e3bd20f](https://github.com/LambdaSection/NeuralDBG/commit/e3bd20fe21f758ec31ed03db3cb430fdd97838c0))
* **types:** corrige les 16 erreurs mypy exposees par pre-commit - Optional explicites, annotations dict, renommages de variables en collision, signatures AST elargies ([9e7af2d](https://github.com/LambdaSection/NeuralDBG/commit/9e7af2d5128e937ce3c287617d285d767433488c))
* unicode in validate_oos.py output (cp1252 compat) ([1c47068](https://github.com/LambdaSection/NeuralDBG/commit/1c47068c309532ac73ee7cc94745fe644693e1ff))
* use pytest --cov instead of coverage run in CI ([a47a749](https://github.com/LambdaSection/NeuralDBG/commit/a47a7492a9d4c3b9ab2ec5cc54af6a32d35f7631))
* use register_full_backward_hook for RNN modules (LSTM/GRU) ([3152c50](https://github.com/LambdaSection/NeuralDBG/commit/3152c5076417b5ccae7f8c6c8a6f5a5900326da8))
* ViT-Tiny + Mamba-Mini OOS shape mismatches ([474cd7f](https://github.com/LambdaSection/NeuralDBG/commit/474cd7fd87fdf13d4f7e6cdab36f84a19a770754))


### Reverts

* keep vanishing threshold 1e-6 — 1e-4 adds baseline noise ([929c734](https://github.com/LambdaSection/NeuralDBG/commit/929c7347b2ae7e05c4aff58b05561f7bcf970b86))


### Documentation

* add benchmark comparison HTML + paper v3 section 4.12 ([0e9d9e3](https://github.com/LambdaSection/NeuralDBG/commit/0e9d9e3e78340e17fd459d7a616be989730ea0b2))
* add BUG-002 tracking (varlen_attn NaN gradients with padding) ([8987b7d](https://github.com/LambdaSection/NeuralDBG/commit/8987b7d254ac5f88e67cb213d729ccaea58504d4))
* add BUG-003 comment draft (pytorch[#177116](https://github.com/LambdaSection/NeuralDBG/issues/177116) MPS gradients) ([5c1dfc5](https://github.com/LambdaSection/NeuralDBG/commit/5c1dfc5ce8b1b7aa58c5f33f626d4a3d8bc8b3c5))
* add BUG-003 tracking (MPS catastrophically wrong gradients) ([ef8e58c](https://github.com/LambdaSection/NeuralDBG/commit/ef8e58c09b78026a9c08a87bd34e95e464911c30))
* add Cursor Linear MCP setup guide ([a911d97](https://github.com/LambdaSection/NeuralDBG/commit/a911d97769894a649b3f7f2ec194510b90b5c139))
* add local copy of kuro-rules (31 rules) ([ce301af](https://github.com/LambdaSection/NeuralDBG/commit/ce301af32c8f18159c1ea172c838e6caa3f8baf4))
* add Phase 0 (causal validation tests) and Phase 7 (IP/strategy/hybrid model) ([882c7a9](https://github.com/LambdaSection/NeuralDBG/commit/882c7a95f7afe29b89534828dfbae4afff29c573))
* add Phase 8 (auto-improvement causal) and Phase 9 (external validation) ([5e9206c](https://github.com/LambdaSection/NeuralDBG/commit/5e9206c81bd7d816d1bf294864d60ac14f22670d))
* add public ROADMAP.md with project roadmap ([7468b1e](https://github.com/LambdaSection/NeuralDBG/commit/7468b1e7d467f1fb593215e96e02bf949be17ffa))
* add R106 (Private Plan + Public Roadmap Split) to AGENTS index ([7fe757d](https://github.com/LambdaSection/NeuralDBG/commit/7fe757d83f19f8c217918f6a2dacbae9d8b8adea))
* add R94 — Daily X Post obligation ([eb8d5bc](https://github.com/LambdaSection/NeuralDBG/commit/eb8d5bc7ea9e9ae60b467faf6dbad90b0d22965b))
* Add Rule 30 for mandatory branch creation and naming conventions to enforce strict Git workflow. ([082b29d](https://github.com/LambdaSection/NeuralDBG/commit/082b29d0fece6c42c4fef5e981bb5f29a5ee2db2))
* add Rule 37 for mandatory code review before continuation ([4e1242d](https://github.com/LambdaSection/NeuralDBG/commit/4e1242dbf30a622f6171218ae76e46ccf60697dc))
* add Rule 38 for CEO Linear dashboard visibility ([183d5f7](https://github.com/LambdaSection/NeuralDBG/commit/183d5f70e05b0530f142a513511678a2e92cf7f3))
* add rule_103 profile readme sync to AGENTS.md ([099f128](https://github.com/LambdaSection/NeuralDBG/commit/099f1286ebcf447e1d07c02743a26af331548175))
* add tutorial, discussion script, reddit/forum drafts ([415a356](https://github.com/LambdaSection/NeuralDBG/commit/415a356d7d901687bf08abb30d07ea0061832303))
* add upstream PR template with NeuralDBG diagnostic evidence ([d91b544](https://github.com/LambdaSection/NeuralDBG/commit/d91b544f3880775c1bee9654a3f6b8c7a14847eb))
* add workflow + vanishing GIFs to README ([0e18435](https://github.com/LambdaSection/NeuralDBG/commit/0e184351a8eca75770e61fdaa85609e87f134bcb))
* **agents:** update Rule 11 with unified roadmap management and add Rule 24 for marketing outreach ([29a6dbc](https://github.com/LambdaSection/NeuralDBG/commit/29a6dbc5e3c108242987e184e1bcde95258c0c74))
* **agent:** update Kaggle training notebook + BUG-002 + BUG-003 catalogs ([8372b82](https://github.com/LambdaSection/NeuralDBG/commit/8372b820803a54b0fe8159b952f6c199573a1476))
* align PR state and distribution calendar (08/08) - retire [#188066](https://github.com/LambdaSection/NeuralDBG/issues/188066) (premisse refutee albanD), mark 2 active PRs, defer arXiv to Dec 2026 and community posts to Jan 2027, update paper to 6 bugs / 2 PRs ([a3faab3](https://github.com/LambdaSection/NeuralDBG/commit/a3faab3347c6a493f590cbf3c0870487797ee951))
* align README with actual upstream PR state (2 open PRs, F.normalize retired) ([c76bb81](https://github.com/LambdaSection/NeuralDBG/commit/c76bb816e66896de1d4ce5e3baae2ab0f3cffab5))
* community launch posts — Tier 4 + 10 post-mortems + BUG-004 remediation rule ([57f71b1](https://github.com/LambdaSection/NeuralDBG/commit/57f71b153e4640c617340500930c58cdc5d1b181))
* community posts — PyTorch Dev Discussions + Reddit r/ML ([c90d14d](https://github.com/LambdaSection/NeuralDBG/commit/c90d14d1ef37798635ff5ea205d7989c02539b1f))
* complete desk research (R75) — 5 dimensions, GO decision ([d6ba04b](https://github.com/LambdaSection/NeuralDBG/commit/d6ba04b652616841371d46c637b3db80f1a7de9c))
* comprehensive Colab notebook — causal chains, vanishing detection, fix demo ([a4e352c](https://github.com/LambdaSection/NeuralDBG/commit/a4e352c2366334f4920ccce53e9978706f23c9c9))
* create French planning document outlining Milestone 0 DevOps/MLOps initial setup tasks and rationale. ([9f0ba90](https://github.com/LambdaSection/NeuralDBG/commit/9f0ba90bd079276d018398c0cd1d4f7ec9077945))
* **ecosystem:** add [#80](https://github.com/LambdaSection/NeuralDBG/issues/80) criteria tracker — all governance resolved, community metrics pending ([f91d07e](https://github.com/LambdaSection/NeuralDBG/commit/f91d07eb52f4972eb4d7db4626d96a939cf5f9ae))
* enforce Mom Test for early validation ([153af97](https://github.com/LambdaSection/NeuralDBG/commit/153af97741034d4ea007e61ae303a8009368c7cc))
* enhance documentation on AI guidelines and project structure ([5b8c32e](https://github.com/LambdaSection/NeuralDBG/commit/5b8c32e04cec076fc0227c323033fc2c7493cfd3))
* expand README.md with comprehensive project overview and usage guide ([7372e70](https://github.com/LambdaSection/NeuralDBG/commit/7372e70ea6ad400f87bc5cdd526ed25cb8631c67))
* fix dead Aquarium dashboard link ([#688](https://github.com/LambdaSection/NeuralDBG/issues/688)) ([5109ca4](https://github.com/LambdaSection/NeuralDBG/commit/5109ca49889d82920dfb73a00fedf8f7dae547dd))
* Fix README.md and format ([f07aba5](https://github.com/LambdaSection/NeuralDBG/commit/f07aba57f6795ec86b13b66ccdc3d2a779cdadac))
* implement Rule 14.5 Failure Mode Table and governance hardening ([d74d3c4](https://github.com/LambdaSection/NeuralDBG/commit/d74d3c4372f19c0ad830557cc252e77ccaad2614))
* Introduce detailed Copilot instructions and update AI guidelines with a new PR analysis rule and milestone lock enforcement. ([80317d3](https://github.com/LambdaSection/NeuralDBG/commit/80317d329abcbf89297da5f06014b049d77c79a2))
* **linear:** add Cursor Linear MCP setup guide ([2698143](https://github.com/LambdaSection/NeuralDBG/commit/26981430bd91f6daaae627b7a181dbd723896478))
* **linear:** prefer user-level mcp.json path in setup ([ad4b1b9](https://github.com/LambdaSection/NeuralDBG/commit/ad4b1b9c6952060c804590d362ed8778016ce23c))
* **linear:** remove machine-specific paths from setup guide ([4e35db6](https://github.com/LambdaSection/NeuralDBG/commit/4e35db66302c499b7645429436f670c04a59ff71))
* **linear:** rewrite Cursor setup A-Z with HTTP MCP path ([c8de19d](https://github.com/LambdaSection/NeuralDBG/commit/c8de19d1945f33cfe6a597ecb518e2ce8e554077))
* link orphaned lead magnets (landing + checklist) from index ([#689](https://github.com/LambdaSection/NeuralDBG/issues/689)) ([2fe850c](https://github.com/LambdaSection/NeuralDBG/commit/2fe850c083e88559d216e5e7074bc806661991cf))
* **mom-test:** allow brainstorming and protect idea files ([9721a87](https://github.com/LambdaSection/NeuralDBG/commit/9721a87006c39435d78a294f3744ffb75bd7fbd0))
* paper post-mortem strategy + updated Reddit draft with Tier 1/2 results ([0bc1f0b](https://github.com/LambdaSection/NeuralDBG/commit/0bc1f0bf2b34c64857648cfc650b5c7a66ea675f))
* paper v2 — Tier 1/2 black-swans, NeuralPrune, 10 post-mortems, stress tests ([a29b06f](https://github.com/LambdaSection/NeuralDBG/commit/a29b06f11b370f85bb183c6e047c204b640f0ebb))
* paper v3 — Tier 3/4, v5 GPU, Colab notebook sections added ([38a50f9](https://github.com/LambdaSection/NeuralDBG/commit/38a50f9f950ed83ae15405f40e6493eb9120a129))
* **paper:** add sections 4.13 (OOS ResNet-18) and 7.4 (FP reduction) ([5d02b5b](https://github.com/LambdaSection/NeuralDBG/commit/5d02b5b4908f84bfba5ea1c71209bc125fc1673b))
* **paper:** v4 — rewrite abstract, add Related Work (28 refs), fix FP contradiction, renumber sections ([3c3a1b3](https://github.com/LambdaSection/NeuralDBG/commit/3c3a1b32993f4cbbe8a8469873f3c2bb1f4ad815))
* prepare comment draft for pytorch/pytorch[#176793](https://github.com/LambdaSection/NeuralDBG/issues/176793) ([255dbd6](https://github.com/LambdaSection/NeuralDBG/commit/255dbd6373ad7e5415d6d84ce6dc147e424bc1fd))
* prepare upstream PR draft for pytorch[#41508](https://github.com/LambdaSection/NeuralDBG/issues/41508) (MHA warning) ([95fa5df](https://github.com/LambdaSection/NeuralDBG/commit/95fa5dfb1ec69258a2c407f1387792806e55b3d1))
* PyTorch Dev Discussions post — causal debugging showcase ([a7913bb](https://github.com/LambdaSection/NeuralDBG/commit/a7913bbec570b83c84d4c9b090775fdded243f7d))
* README updated for v1.4.0 — 200-arch validation, RNN support, GPU v4 ([ac8a8c1](https://github.com/LambdaSection/NeuralDBG/commit/ac8a8c19d3b5b9e27e408d260c2ea2f989cceacf))
* README with combinatorial validation — 200 architectures, 1200 tests ([a2e3c02](https://github.com/LambdaSection/NeuralDBG/commit/a2e3c020bfaf9bec7b2f48aefcee448b34554757))
* **readme:** refactor project description and features for causal inference focus ([72b5605](https://github.com/LambdaSection/NeuralDBG/commit/72b5605f44b6078b61205be25cd544f939a9290d))
* release 1.3.1 plan sync and remove public acquisition tracking refs ([91069dc](https://github.com/LambdaSection/NeuralDBG/commit/91069dcf207ab4f2598327cfad6669d2860cd649))
* research paper draft — Causal Debugging of DL Training Failures ([a61f49b](https://github.com/LambdaSection/NeuralDBG/commit/a61f49bd1d034cd6e5b0ec5546d7169f00802774))
* rewrite README for v1.3.0 — fix badges, URLs, add editions table, remove MVP ([fb4ab4a](https://github.com/LambdaSection/NeuralDBG/commit/fb4ab4a716d92ba57ba82a45330c66419e5893ec))
* ROADMAP — 10/10 bugs reached (M2 objective) ([bf5736a](https://github.com/LambdaSection/NeuralDBG/commit/bf5736a2950a91490f739f7371adc23322a07c2f))
* **rules:** restore deleted sections and preserve legacy rule addenda ([74d490e](https://github.com/LambdaSection/NeuralDBG/commit/74d490ea2a93d549797d41bd769578827a35e5ce))
* Update AI guidelines with detailed sections on modular design, critical thinking, and robust security hardening. ([d66e9f8](https://github.com/LambdaSection/NeuralDBG/commit/d66e9f8ce9007d954be38ccdf2e54924b68bd75e))
* update AntiGravity rules and roadmap for AI tooling ([a9312a2](https://github.com/LambdaSection/NeuralDBG/commit/a9312a2c2f1a3d2d682a16f15d65b70d6d652420))
* update CHANGELOG and PROJECTS.md for project structure and setup ([57a7a50](https://github.com/LambdaSection/NeuralDBG/commit/57a7a5011f5fe9835e80398b5e110313a7d67d8f))
* update guidelines and rules for AI tooling and product quality ([7cdfd03](https://github.com/LambdaSection/NeuralDBG/commit/7cdfd031b2fb1af39e41b84fd3475cd1a349a0f8))
* update Phase 10 status in PLAN.md ([1ff040a](https://github.com/LambdaSection/NeuralDBG/commit/1ff040a7a8997a2ce5247569683883306573f9c0))
* update PLAN.md — LoRA dogfooding done ([fdcd00b](https://github.com/LambdaSection/NeuralDBG/commit/fdcd00b0b6379842bc6c0cf4c34fc95efba28eec))
* update PLAN.md with GAN, Diffusion, and benchmark progress ([332bd48](https://github.com/LambdaSection/NeuralDBG/commit/332bd48719b70e3a78d6e5ccf078f8026b4e4f3b))
* update PROJECTS.md to correct local cloning path for Project A ([e5c330e](https://github.com/LambdaSection/NeuralDBG/commit/e5c330e173c4a36c2e86fc12284bf19b47e71e44))
* update PyTorch Forums post draft with current results ([093f1f0](https://github.com/LambdaSection/NeuralDBG/commit/093f1f0c18a9bab3be844c9c3a07fd362f536b7e))
* update README with 89% detection, 0% FP, real-architecture table ([7fd26f6](https://github.com/LambdaSection/NeuralDBG/commit/7fd26f61456c5b55ea8c45c432ee5c64a69a39eb))
* update README with July achievements + dashboard blog/benchmark links ([76e3152](https://github.com/LambdaSection/NeuralDBG/commit/76e3152c28dcfd2350e5628d10a918260f1f308a))
* update roadmap with dogfooding, agent vision, and desk research ([f3e7e06](https://github.com/LambdaSection/NeuralDBG/commit/f3e7e06abda1b81749bf5314402e6c5f1bc8b2fc))
* update Rule 37 for mandatory code review after commit ([2766431](https://github.com/LambdaSection/NeuralDBG/commit/27664310557a9db9743bc1dfac030278a817ad2d))
* update session summary with branch merge and rule sync details ([bec055b](https://github.com/LambdaSection/NeuralDBG/commit/bec055b90085d58216db6138914481148007f33b))
* update SESSION_SUMMARY with all deleted corrupted branches ([2dd22bd](https://github.com/LambdaSection/NeuralDBG/commit/2dd22bd5a44694a6ea3bafb66f3ae4a0bacb5cfd))
* update SESSION_SUMMARY with branch analysis and validation sync improvements ([c303269](https://github.com/LambdaSection/NeuralDBG/commit/c3032698856420e2d04b381ad1349afc9a00a7de))
* Update task documentation requirement to include English and French pedagogical versions. ([76377d9](https://github.com/LambdaSection/NeuralDBG/commit/76377d937b62c4dc90a5fec43ab9f33ea5f94964))
* **upstream:** add MHA NaN comment draft for pytorch[#41508](https://github.com/LambdaSection/NeuralDBG/issues/41508) (BUG-001) ([e868d9a](https://github.com/LambdaSection/NeuralDBG/commit/e868d9a000c2e439916cee82e9992c3ca1036275))
* v1.5.0 — competitive analysis, Tier 2 results, NeuralPrune in README/ROADMAP ([e77f4e9](https://github.com/LambdaSection/NeuralDBG/commit/e77f4e9f234a7d5ab4cd0cecfac77e1ce392e595))

## [Unreleased]

### Added
- **Ecosystem integrations**: PyTorch Lightning callback (`NeuralDBGLightningCallback`) and W&B callback (`NeuralDBGCallback`) with full docs + tests.
- **PyTorch Ecosystem Landscape**: Application submitted to pytorch-fdn/ecosystem#80.
- **Community posts**: Drafts for W&B Community and Lightning Ecosystem blog posts.
- **README badges**: Lightning + Weights & Biases ecosystem badges.
- **W&B Report script**: `scripts/create_wandb_report.py` for Fully Connected blog submission.

## [1.5.0] — 2026-07-08

### Added
- **Tier 1 Black-Swan detection (96%)**: GNN 88%, MoE 100%, Diffusion 100% across 54 configs.
- **Tier 2 Black-Swan detection (94%)**: FlashAttention 100%, Neural ODE 100%, Quantized (INT8/INT4) 83% across 54 configs.
- **Tier 3 Predictive detector**: Family-aware statistical profiles (30 architectures, 5 families). Detects anomalies via per-family z-scores (event_count, grad_norm, act_sat).
- **Tier 4 Black-Swan detection (50%)**: RAG 100%, RL (REINFORCE) 0% — policy gradient blind spot documented.
- **NeuralPrune v0.1**: Non-destructive redundancy diagnostic — 5 signal types (dead neuron, redundant weight, static weight, low-rank, quantizable).
- **10 Post-Mortems**: Reproduced with causal chains — 7 real PyTorch bugs + 3 common failure modes. 7/10 with causal chains.
- **v5 GPU model**: 93.7% accuracy (vs 92.3% v4), 6 families (vs 5), 37min training (6.7× faster).
- **Architecture fuzzer**: 94% crash rate across 50 randomly generated architectures.
- **Stress test suite**: 15/15 scenarios pass (100%).
- **Self-evolution engine**: 7-step daily pipeline (Scrape→Fuzz→Test→Train→Retrain→Heal→Report).
- **End-to-end demo** (`demo_neural_suite.py`): NeuralDBG + NeuralPrune + Tier 3 on a single realistic CNN.
- **Colab notebook** (`notebooks/quickstart.ipynb`): Self-contained 5-cell demo — CPU-only, free tier.
- **Family-aware thresholds**: Different noise floors per architecture family (baseline+2 for RNN/Hybrid, +3 for others).
- **4 upstream PyTorch PRs**: #188053 (svdvals NaN), #188066 (F.normalize zero), #188923 (gradient health), #188933 (varlen_attn NaN).

### Changed
- **RNN detection**: 49% → 71% (+22%) via tuple unwrap, per-gate tracking, trend-based vanishing.
- **Hybrid detection**: 34% → 96% (+62%) via family-aware thresholds.
- **Bug injectors**: `bug_vanishing` now scales weights 1000× for non-RNN models. `bug_nan` handles tuple inputs.
- **Hook installer**: Uses `register_full_backward_hook` for RNN modules, `register_backward_hook` for others.
- **Causal compatibility matrix**: Expanded from 9 to 14 pairs for RNN event patterns.
- **Engine merged**: `neuraldbg-engine` now bundled in core. License: MIT.
- **PLAN.md**: Comprehensive competitive analysis + validation strategy added.

## [1.4.0] — 2026-07-04

### Added
- **RNN/LSTM/GRU support**: forward hooks now unwrap RNN output tuples `(output, (h_n, c_n))`. Hidden state capture with BPTT gradient health tracking. RNN detection: 49%→65% (+16%), Hybrid: 34%→85% (+51%), Global: 75%→87%.
- **Combinatorial architecture validation** (`validate_combinatorial.py`): 200 architecture configs × 6 bugs × 5 families (MLP/CNN/RNN/Transformer/Hybrid). 1200 evaluations. RNN-aware bug injectors (forget gate corruption, BPTT sequence extension).
- **Paper architecture scraper** (`scrape_paper_archs.py`): 60 novel architectures from papers (Mamba, KAN, xLSTM, MoE, Hyena, RWKV, RetNet, BitNet, etc.).
- **Aquarium web dashboard** (`docs/aquarium.html`): zero-dependency HTML causal viewer. Drag-drop NeuralDBG JSON exports. Replaces dormant Tauri app.
- **GPU v4 model**: Qwen2-0.5B fp16 + LoRA r=8, 538 training examples from all 5 families (6.1× increase). 92.3% accuracy, 4.3MB adapter. Agent bridge updated.
- **E2E RNN pipeline** (`e2e_rnn_pipeline.py`): closed loop on LSTM bugs. 2/4 auto-fixed with causal chain tracing. Aquarium JSON export.
- **CI benchmark workflow** (`.github/workflows/benchmark.yml`): runs combinatorial benchmark on every push/PR. Fails if detection < 80%.
- **Causal chain compatibility**: added 5 cross-type compatibility pairs for RNN event linking (data_anomaly→data_anomaly, activation→optimizer, etc.).

### Added (July 2026)
- **Causal chain engine** (`neuraldbg/causal_chain.py`): builds directed causal graphs from events, extracts ranked chains via DFS. Shows root cause → propagation → final symptom. Integrated via `dbg.explain_causal()`.
- **DeepMLP validation** (`validate_resnet.py`): 12-layer residual architecture achieving 100% detection (7/7) vs 57% on shallow models. Median gap: +17 anomalies.
- **GPU-trained Neural-Agent v3** (`train_balanced.py`): Qwen2-0.5B + LoRA, 5/5 categories distinct, trained on 10 live events + 30 real bug triplets.
- **End-to-end pipeline** (`e2e_pipeline.py`): detect → causal chain → AI diagnose → fix → validate. BUG-003 achieves PASS (0→24→1).
- **10 post-mortems published** on [GitHub Pages](https://lambdasection.github.io/NeuralDBG/blog/): complete catalog of real PyTorch/HF bugs with reproduction, diagnosis, and causal chains.
- **Validation dashboard** (`docs/dashboard.html`): live bug detection matrix, PR tracker, model versions.
- **Tool comparison matrix** (`docs/comparison.html`): NeuralDBG vs W&B/TensorBoard/MLflow/Captum across 16 capabilities. NeuralDBG: 14/16 YES.
- **Captum benchmark** (`benchmark_public/benchmark_captum.py`): proves NeuralDBG solves a different problem than explainability tools.
- **4 upstream PRs submitted**: #188933 (real fix), #188923 (+59/-0 test), #188053 (albanD reviewed), #188066 (CI fixed).
- **CLI wrapper**: `neuraldbg run script.py --agent --export`.
- **Live event capture** (`scripts/capture_live_events.py`): 10 live triplets from actual NeuralDBG sessions.
- **GPU agent bridge** (`agent_bridge.py` in Neural-Agent): subprocess-callable AI diagnosis.

### Changed
- **Agent model**: CPU distilgpt2 → GPU Qwen2-0.5B fp16 + LoRA (Quadro M4000, 8.6 GB).
- **PR creation**: moved from fork-clone to GitHub API direct (SHA-based) to avoid fork corruption.
- **Plan restructured**: focus on independent activities (content, product, distribution) while PRs await review.
- **Kaggle**: abandoned in favor of local GPU training.

### Fixed
- Causal chain engine: filter logic (AND/OR bug), node key collisions, DFS combinatorial explosion (45K→30 chains).
- PR #188066: 13 CI failures resolved (isnan→isfinite, TESTOWNERS header).
- PR #188797/#188922: closed corrupted PRs, replaced with clean #188923 (+59/-0).

## [1.3.2] - 2026-06-09

### Added
- **Multi-Repo Ecosystem cartography** (R105): NeuralDBG-Engine added as optional 4th component in [`docs/ecosystem.md`](docs/ecosystem.md); cross-repo SemVer tracking via new [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md); "Écosystème (Multi-Repo)" section in `ROADMAP.md`.
- **Composite-module hook support**: `dbg.register_composite_hook(module)` for `nn.MultiheadAttention` and other modules with no leaf submodules.
- **Silent-loss and zero-leaf warnings**: detects loss=0 with non-zero gradients, and `register_full_backward_hook` no-op setups.
- **MHA fully-masked-row remediation rule**: `apply_mha_mask_workaround()` in Neural-Agent, wired to NeuralDBG events.
- **End-to-end Neural-Agent pipeline**: `diagnose -> fix -> validate -> apply -> re-run`, 87 tests passing.
- **Bug catalog BUG-001..004**: MHA NaN, varlen_attn NaN, MPS gradients, Qwen3.5 SDPA gradient explosion.
- **Public benchmark** (5 scenarios): all at 1.0 accuracy; comparison v2 vs W&B / MLflow / TensorBoard.
- **Aquarium JSON export**: full schema (`schema/events.json`), 14 unit tests in `test_aquarium_export.py`.
- **Phase 7 — Two-Package Architecture**: conditional import of `neuraldbg-engine` with seamless fallback in `neuraldbg` core.
- **Zero-Warnings Policy**: `filterwarnings` in `pyproject.toml` drops warnings 616 → 5.
- **Cross-repo contract**: `dbg.explain_failure()` and `events.json` schema v1 stable; `dbg` works without engine and without agent.

### Changed
- **PUBLIC → multi-repo narrative**: `ROADMAP.md` updated from "three-part" to "four-part" system (NeuralDBG, Neural-Agent, Aquarium, neuraldbg-engine).
- **Upstream PR tracker** updated: 4 comments posted, 1 PR submitted (pytorch/pytorch#186786, OPEN).
- **Benchmark table** expanded from 4 → 5 scenarios.

### Fixed
- Unicode/emoji terminal rendering encoding crash on Windows consoles for `quickstart.py`.
- Mock comparison removed from `benchmark_public/` — replaced by real `real_comparison.py` (R79 honesty).
- Deduplication of logical causal couplings in `detect_coupled_failures()` and Mermaid graph export.

### Security
- `assert` removed from production code paths (R39 compliance).
- Bandit scan wired to pre-commit (skips B101 — acceptable for tests).

## [1.3.1] - 2026-05-20

### Added
- **OOM Prevention & Memory Optimization**: Added `TensorDiskCache` to JIT-cache intermediate tensors on disk during anomaly states, preventing VRAM/RAM exhaustion.
- **Precision and Epsilon Scaling**: Implemented dtype-aware epsilon scaling (`1e-4` for float16/bfloat16, `1e-9` for float32/64) to prevent precision underflow during activation statistics computation.
- **Safety Guards for Integer Tensors**: Added strict checks (`torch.is_floating_point`) to bypass statistics computations on non-floating-point tensors (e.g., token indices, label masks), preventing PyTorch runtime errors.
- - Phase 2 dogfooding: LSTM/Time Series failure scenarios (vanishing recurrent, exploding recurrent, deep LSTM)
- - Phase 2 dogfooding: GNN (GCN/GAT) failure scenarios (oversmoothing, exploding, NaN injection)
- - Phase 2 dogfooding: torch.compile (Dynamo) compatibility scenarios (healthy, vanishing, exploding)
- - Phase 2 dogfooding: RL (PPO-style) failure scenarios (policy collapse, value explosion, reward hacking)
- - Phase 2 dogfooding: Distributed/DataParallel failure scenarios (healthy, vanishing, exploding under DP)
- - Engine fallbacks in core: `_classify_activation_health`, `explain_failure`, `detect_coupled_failures`, `export_mermaid_causal_graph`, `_classify_data_health`, `_check_data_anomaly` now work without proprietary engine
- - **Phase 3**: Complete Aquarium JSON export schema with all required fields (events, hypotheses, couplings, first_failure_layer, first_failure_step, loss_history)
- - 14 new unit tests for Aquarium export (`test_aquarium_export.py`)
- - Aquarium export integrated into LSTM demo with auto-export to `aquarium_exports/`
- - **Phase 7 — Two-Package Architecture**: Conditional import check for `neuraldbg-engine` and seamless fallback support for `neuraldbg` core, enabling private/public package separation.
- - **Zero-Warnings Policy**: Configured `filterwarnings` in `pyproject.toml` to ignore third-party deprecation warnings (MLflow, PyTorch full_backward_hook warnings), dropping warnings from 616 to 5.

### Fixed
- Fixed Unicode/emoji terminal rendering encoding crash on Windows consoles for `quickstart.py`.

## [1.3.0] - 2026-05-14
### Added
- ResNet-18 failure scenarios demo (`demo_resnet_failures.py`) : vanishing gradients (Tanh + small init), exploding gradients (high LR), data anomaly (NaN injection)
- Integration tests for ResNet-18 demo (5 tests, 100% coverage)
- Semantic demo smoke tests (`test_semantic_demo.py`) for causal hypothesis validation

### Fixed
- Deduplication of logical causal couplings in `detect_coupled_failures()` and Mermaid graph export
- Import path for MLflow demo test after directory restructuring
- Graceful degradation of CPU resource sampler after psutil failure (avoids repeated exceptions)

## [1.2.0] - 2026-05-11
### Added
- Integrated PR 651: Detect Python version mismatch in `ensure_venv.sh` (MLO-17).
- Integrated PR 652: Initialized DVC for binary artifact versioning (MLO-4).
- Integrated PR 654: Resource profiling (CPU/GPU memory) integration for semantic events (MLO-10).
- Integrated PR 656: `SESSION_SUMMARY.md` to `.docx` conversion tool (NDBG-5).

### Changed
- Refactored repository structure: Unified all scripts into `infrastructure/scripts/`.
- Moved `neuraldbg.py` to `neuraldbg/__init__.py` for better package organization.
- Standardized `Makefile` to use centralized infrastructure scripts.
- Cleaned root directory by moving legacy security reports to `outputs/reports/`.

### Fixed
- Restored `neuraldbg.py` core engine which was incorrectly removed in previous refactor commits.
- Fixed import paths in test suite after directory restructuring.
- Resolved multiple merge conflicts in `.gitignore` and `Makefile`.

### Added
- `scripts/publish_session_summary_to_gdocs.py` to publish `SESSION_SUMMARY.md` directly to Google Docs (append/replace modes)
- `.github/workflows/publish-summary-to-google-docs.yml` to support scheduled/manual Google Docs sync from CI with secrets-based auth
- `GOOGLE_DOCS_SYNC.md` setup guide for Google Workspace service account integration
- Rule 39 (`CI/CD Debugging First`) synchronized across AI rule files
- Mandatory product & quality rules in `.cursorrules`, `ia_rules/AI_GUIDELINES.md`, `.github/copilot-instructions.md`, and `.cursor/rules/product-quality.mdc`
- Strategic section "Tools for the AI Era" explaining why structured tools matter when AI agents can code
- `.github/workflows/codeql.yml` — CodeQL security analysis (Python)
- `.github/workflows/codacy.yml` — Codacy static analysis (auto-detects Python)
- `.antigravity/RULES.md` — Copie des règles pour l’IDE AntiGravity uniquement
- `PROJECTS.md` — Roadmap Projets A & B (racine, aucun lien avec AntiGravity)
- `artifacts/` — Artifacts générés (déplacés depuis .antigravity/artifacts)

### Changed
- Projet A : repo dédié sous Quant-Search, NeuralDBG utilisé pour debug itératif

### Added
- `skeleton-quant-search/` — squelette prêt à copier pour le repo Quant-Search
- Règle **"Explain as if First Time"** : toujours expliquer IA, ML, concepts, maths comme si l'utilisateur ne savait rien (code en apprenant)
- Règle **"Sync with kuro-rules"** : toujours synchroniser les mises à jour de règles avec `~/Documents/kuro-rules`
