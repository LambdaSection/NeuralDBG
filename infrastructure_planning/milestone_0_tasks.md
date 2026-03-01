# DevOps / MLOps Tasks — Milestone 0 (Initial Setup)

---

## 🧠 Pedagogical Brief for the CEO (Why Delegate?)

**Q: "Can I do all this myself? Is it worth delegating if it takes a month?"**

**Yes, you *can* do it yourself.** Technically, writing a GitHub Action takes a few hours. 
**HOWEVER:** Every context switch from "CEO/ML Researcher" to "DevOps Plumber" drains your cognitive load. If you spend 20 hours fixing a Docker volume bug or a CI pipeline, that is 20 hours you did NOT spend talking to users (Mom Test) or improving your core ML algorithm. Delegating allows you to stay in the "Hub" (Core Logic) while your friend builds the "Spokes" (Delivery Pipeline).

### The CI/CD "Gate" Concept Explained

**Q: "If the gate isn't in the cloud, is it easy to commit bad code?"**

**Exactly.** Right now, you have local tools (`pytest`, `bandit`), but they rely on *human discipline*. A tired developer can easily type `git push` without running the tests. A Cloud CI/CD Pipeline is an **unbreakable digital bouncer**.

```text
  [LOCAL WORKFLOW - Vulnerable to Human Error]
  
  Dev Machine           Git Repo
  +----------+         +--------+
  | Bad Code | ------> | MERGED | (Oops! Broken code in master)
  +----------+  push   +--------+
                ^
               Did they run tests? 
               Maybe. Maybe not.

  [CLOUD CI/CD WORKFLOW - Unbreakable Hub]
  
  Dev Machine                       GitHub / Cloud
  +----------+         +-------------------------------------+      +--------+
  | Bad Code | ------> | [CI GATEKEEPER]                     |      |        |
  +----------+  push   | 1. Run tests (Coverage < 60%) ❌ | -/-> | MERGED |
                       | 2. Run bandit (Security fail) ❌ |      |        |
                       +-------------------------------------+      +--------+
                                         |
                                         v
                            [PUSH REJECTED - TRY AGAIN]
```

---

## 🛠️ The 5 Tasks

### 1. Cross-Platform CI/CD with Enforcement Gates (DevOps)
* **Task**: Build a GitHub Actions workflow that automatically runs tests on both Linux and Windows runners. This pipeline must strictly enforce Rule 5 (60% minimum coverage) and Rule 6 (security scans with `bandit` and `safety`).
* **ROI Estimation**: Saves ~2 hours/week of manual testing. Eliminates the risk of merging broken or insecure code.

### 2. Model & Experiment Tracking (MLOps)
* **Task**: Integrate an experiment tracker (like MLflow or Weights & Biases) into the existing `demo_vanishing_gradients.py` and future model scripts. Track loss curves, gradients, and hyper-parameters automatically (remove dependency on `matplotlib` popups).
* **ROI Estimation**: Saves ~3 hours per model iteration parsing logs manually.

### 3. Hermetic Workspaces via Docker (DevOps)
* **Task**: Create a standardized `Dockerfile` and `docker-compose.yml` tailored for PyTorch. Include volumes for models/data to allow fast local development without dependency conflicts.
* **ROI Estimation**: Saves ~4-5 hours of onboarding time for any new contributor or AI agent.

### 4. Binary and Data Versioning (MLOps)
* **Task**: Set up a designated registry or use DVC (Data Version Control) to manage artifacts like `synthetic_data_sample.png` and future trained weights. Remove them from git tracking to prevent repo bloat.
* **ROI Estimation**: Prevents repository size limits. Saves minutes on every `git push/pull`.

### 5. Session Sync Automation Script (DX / DevOps)
* **Task**: Implement a script (e.g. Python using `python-docx`) that automatically converts `SESSION_SUMMARY.md` into a `.docx` or styled PDF document for external communication, as requested in Rule 4.
* **ROI Estimation**: Saves the lead developer ~15 minutes per session (~2 hours/week of admin work).
