# Milestone 50 DevOps/MLOps Tasks

At 50% progress, focusing on experiment tracking and data versioning.

## Tasks

1. **MLflow Event Logging Integration**
   - **Description**: Integrate MLflow to log `SemanticEvent` objects and causal hypotheses automatically.
   - **Acceptance Criteria**: Training loop logs events to a local or remote MLflow server.
   - **ROI**: Saves 1 day per analysis session by providing persistent histories.
   - **Linear Issue**: `infra/MLO-6-mlflow-tracking`

2. **Data Versioning (DVC) for Synthetic Data**
   - **Description**: Setup DVC to track synthetic datasets used for debugging demos.
   - **Acceptance Criteria**: `dvc push` and `dvc pull` work for generated wave forms.
   - **ROI**: Ensures perfect reproducibility and saves hours of re-generating data.
   - **Linear Issue**: `infra/MLO-7-dvc-setup`

3. **Automated Documentation Generation (Sphinx/MkDocs)**
   - **Description**: Setup automated API documentation generation from docstrings.
   - **Acceptance Criteria**: Documentation builds without errors and covers all main methods of `NeuralDbg`.
   - **ROI**: Saves 4 hours/month of manual doc updates.
   - **Linear Issue**: `infra/MLO-11-auto-docs`

4. **Resource Profiling Integration**
   - **Description**: Add lightweight resource profiling (memory/GPU) to semantic events.
   - **Acceptance Criteria**: Events capture memory spikes or GPU utilization dips during failures.
   - **ROI**: Identifies bottlenecks 50% faster.
   - **Linear Issue**: `infra/MLO-12-resource-profiling`

5. **Security Scan Hardening (Safety Check)**
   - **Description**: Integrate `safety` into the CI/CD and pre-commit to check for vulnerable dependencies.
   - **Acceptance Criteria**: CI fails if a high-severity vulnerability is detected.
   - **ROI**: Critical security compliance (Rule 6).
   - **Linear Issue**: `infra/MLO-13-safety-check-integration`
