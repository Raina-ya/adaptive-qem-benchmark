# Adaptive QEM Benchmark — Complete Project Handoff

Prepared for: a new AI assistant with no prior project context  
Prepared on: 2026-11-05 (Asia/Taipei)  
Repository: <https://github.com/Raina-ya/adaptive-qem-benchmark>  
Local working copy used for this handoff:  
`C:\Users\wits\Documents\Codex\2026-10-30\adaptive-qem-benchmark\work\phase-d-publish`

> Important: read `AGENTS.md` before making any change. The scientific contracts,
> evidence labels, fixed-budget accounting rules, and raw-data preservation rules are
> mandatory. Do not infer permission to run quantum hardware, add machine learning, or
> rewrite historical artifacts.

## 1. Project overview

### Purpose

`adaptive-qem-benchmark` is a reproducible, simulator-only research repository for
studying when quantum error mitigation (QEM) helps, wastes resources, becomes
unstable, or fails explicitly under a fixed total shot budget.

The central research premise is that no single QEM method is uniformly useful across
circuits, noise conditions, shot budgets, decision margins, calibration models, and
application tasks. The project therefore compares methods using equal total resource
budgets and reports failures, negative results, unresolved decisions, and mitigation
overhead rather than filtering them away.

The principal methods are:

- unmitigated/raw finite-shot estimation;
- full-assignment-matrix readout mitigation;
- tensor-product, observed-subspace, and matrix-free readout approximations;
- linear zero-noise extrapolation (ZNE) with fold factors 1, 3, and 5;
- combined readout mitigation plus ZNE;
- a frozen rule-based selector with abstention;
- calibration-reuse policies under synthetic drift.

All reported finite-shot results are simulated using Qiskit Aer or deterministic
classical sampling. There are **no hardware-derived rows** in the repository.

### Current Git state

Verified immediately before creating this handoff:

- branch: `main`;
- remote: `origin = https://github.com/Raina-ya/adaptive-qem-benchmark.git`;
- branch state: synchronized with `origin/main`;
- latest implementation commit: `1f2c572 Add calibration amortization and drift benchmark`;
- previous major commit: `e24cab6 Integrate QAOA application case study`;
- implementation worktree was clean before `HANDOFF.md` was created.

After this file is created, `HANDOFF.md` itself is expected to be the only new
uncommitted file unless the user commits it separately.

### Completed work

The repository is complete through the following research tracks:

1. **Core benchmark and contracts**
   - Bell-state and GHZ analytic/noiseless references.
   - Fixed-seed hardware-efficient circuit family.
   - Noise models, unmitigated baseline, resource ledger, readout correction.

2. **ZNE and volumetric benchmark**
   - Unit-equivalent gate folding at factors 1, 3, and 5.
   - Fixed-budget linear ZNE.
   - GHZ and nontrivial-circuit volumetric sweeps.
   - Numeric-tolerant artifact comparison for cross-platform floating-point noise.

3. **Adaptive operating-regime analysis**
   - Cost-aware oracle labels.
   - Frozen non-ML rule selector and deterministic holdout.
   - Synthetic noise-shift evaluation.

4. **Decision reliability and ZNE falsification**
   - Controlled decision margins.
   - Expectation RMSE versus decision-error separation.
   - Probability-level ZNE diagnostics.
   - Matched-cost garbage-folding negative control.
   - Frozen ZNE validity gate.

5. **Combined QEM**
   - Raw, readout, ZNE, and readout-plus-ZNE under one total budget.
   - Factorial interaction analysis.
   - Decision-level combined-QEM evaluation.

6. **Scalable readout mitigation**
   - Widths 2, 4, 6, 8, 10, and 12.
   - Full matrix, tensor-product, observed-subspace, and matrix-free methods.
   - Memory, runtime, support, uncertainty amplification, and failure reporting.
   - Correlated and mismatched calibration stress tests.

7. **MaxCut/QAOA application case study (Phase F)**
   - Exact classical MaxCut on deterministic graphs of widths 4, 6, and 8.
   - Frozen ideal-only `p=1` QAOA candidate pools.
   - Fixed-budget raw/readout/ZNE candidate selection.
   - Top-1 accuracy, top-k recall, simple regret, energy RMSE, unresolved
     probability, and simultaneous-interval decision certification.
   - Paper and README integration completed in commit `e24cab6`.

8. **Calibration amortization and drift (Phase G)**
   - Workloads of 1, 5, 20, and 100 decisions.
   - Slow monotonic, random bounded, abrupt, and asymmetric synthetic drift.
   - Recalibrate every decision, fixed interval, uncertainty-triggered, and never
     recalibrate policies.
   - 60,480 deterministic decision-level rows, all with status `complete`.
   - Target/calibration shots charged explicitly.
   - Implementation and artifacts committed in `1f2c572`.

### In progress / not yet integrated

Phase G code, tests, configuration, raw results, and figures are complete, committed,
and pushed. However:

- Phase G is **not yet described in `README.md`**.
- Phase G is **not yet incorporated into `paper/main.tex` or `paper/main.pdf`**.
- There is no dedicated `docs/calibration_lifetime_protocol.md` or processed Phase G
  summary CSV. These were not required by the Phase G deliverable, but adding a
  concise protocol and deterministic processed summary would improve auditability.
- The current paper date is November 5, 2026 and covers the project through Phase F.

### Blocked or externally dependent items

- Real IBM or other quantum hardware execution is not authorized in the current
  repository contract.
- Machine-learning selectors are not authorized.
- A repository license has not been chosen. `README.md` explicitly states that no
  license exists because licensing intent is unresolved.
- TODO: verify the GitHub Actions result for commit `1f2c572` in the GitHub UI or with
  `gh run list`. The local implementation and verification passed, but this handoff
  did not query the live Actions status.

## 2. Repository / file layout

### Root files

| Path | Purpose |
|---|---|
| `AGENTS.md` | Mandatory repository research contract. Read first. |
| `README.md` | Public overview, principal findings, reproduction commands, limitations. Currently covers through Phase F, not Phase G. |
| `pyproject.toml` | Package metadata, Python version bounds, dependencies, pytest, coverage, and Ruff configuration. |
| `requirements-ci.txt` | Exact dependency versions used by GitHub Actions and the verified local environment. |
| `.gitignore` | Ignores venvs, caches, coverage state, temporary work, LaTeX auxiliaries, logs, and `.env`. |
| `.github/workflows/ci.yml` | Linux/Python 3.12 reproducibility workflow. |
| `HANDOFF.md` | This handoff; expected to be uncommitted when first created. |

### `src/` — implementation

Important modules:

| Path | Role |
|---|---|
| `src/config.py` | Global `ROOT_SEED = 20261102` and deterministic SHA-256-derived child seeds. |
| `src/circuits.py` | Bell/GHZ circuit construction. |
| `src/hardware_efficient_circuits.py` | Deterministic nontrivial circuit family and observables. |
| `src/observables.py`, `src/references.py` | Observable definitions and exact/noiseless references. |
| `src/noise_models.py` | Readout and depolarizing simulator noise specifications. |
| `src/estimators/unmitigated.py` | Raw finite-shot estimator and analytic uncertainty. |
| `src/resource_accounting.py` | Shot splitting and method overhead accounting. |
| `src/mitigation/readout.py` | Full assignment-matrix calibration and deterministic constrained correction. |
| `src/mitigation/readout_tensor.py` | Tensor-product readout approximation. |
| `src/mitigation/readout_subspace.py` | Observed-subspace correction. |
| `src/mitigation/readout_matrix_free.py` | Matrix-free support correction. |
| `src/mitigation/folding.py` | Unit-equivalent gate folding. |
| `src/mitigation/garbage_folding.py` | Matched-cost invalid-folding negative control. |
| `src/mitigation/zne.py` | Linear ZNE fit and diagnostics. |
| `src/mitigation/combined.py` | Readout-plus-ZNE accounting and estimator flow. |
| `src/metrics.py`, `src/decision_metrics.py` | Bias/variance/RMSE and decision-error/Wilson metrics. |
| `src/selector.py`, `src/regime_map.py` | Frozen rule selector and operating-regime labels. |
| `src/zne_validity.py`, `src/zne_validity_gate.py` | ZNE diagnostic and cell-level falsification criteria. |
| `src/readout_scalability_metrics.py`, `src/readout_mismatch.py` | Scalable readout resource model and mismatch utilities. |
| `src/applications/qaoa_selection.py` | Exact MaxCut, QAOA statevectors/candidates, runner logic, application metrics, certification. |
| `src/generate_qaoa_selection.py` | Expensive full Phase F artifact generator and plotting. |
| `src/calibration_drift.py` | Complete Phase G model, generator, summary helper, and figures. |
| `src/run_with_log.py` | Wrapper that runs a module and writes a compact JSON provenance record under `results/runs/`. |
| `src/verify_generated_artifacts.py` | Isolated artifact regeneration and semantic comparator. See warnings below before running it. |

Generator modules generally expose `main()` and can be invoked with
`python -m <module>`. Do not assume every generator is cheap.

### `tests/` — contracts and regression tests

Tests cover:

- analytic Bell/GHZ references and Qiskit bit ordering;
- deterministic seeds and candidate regeneration;
- shot-budget conservation and raw-count reconstruction;
- readout conditioning and cross-platform-stable correction;
- gate-folding ideal equivalence;
- ZNE fits, negative controls, probability diagnostics, and validity gates;
- selector/holdout boundaries;
- scalable readout methods and committed artifact coverage;
- exact MaxCut/QAOA selection and certification;
- Phase G drift matrices, policies, common-random-number pairing, cost accounting,
  deterministic generation, and committed artifacts;
- run-log schema and semantic artifact comparison.

Relevant newest tests:

- `tests/test_qaoa_selection.py`
- `tests/test_qaoa_selection_runner.py`
- `tests/test_qaoa_selection_artifacts.py`
- `tests/test_calibration_drift.py`
- `tests/test_verify_generated_artifacts.py`

### `experiments/`

- `experiments/configs/`: frozen experiment contracts.
- `experiments/manifests/`: row-level planned grids and immutable candidate pools.

Key configurations:

- `volumetric_v1.yaml`: GHZ width/depth/noise/shot grid.
- `hardware_efficient_v1.yaml`: nontrivial circuit grid.
- `extended_benchmark_v1.yaml`: global evidence/status/statistical rules.
- `decision_benchmark_v1.yaml`: controlled decision margins.
- `combined_qem_v1.yaml`: combined-QEM cells and allocation policies.
- `readout_scalability_v1.yaml`: widths 2–12 and resource gates.
- `qaoa_selection_v1.yaml`: Phase F candidate, budget, tie, and certification rules.
- `calibration_lifetime_v1.yaml`: Phase G workload, drift, policy, cost, and scope rules.

The `.yaml` extension does not guarantee YAML syntax. Several files, including the
QAOA and calibration-lifetime configurations, are JSON-compatible documents and are
loaded with Python's standard `json` module. Do not add PyYAML unless genuinely needed.

### `results/`

- `results/raw/`: immutable repeat/decision-level rows. Approximately 291 MiB across
  17 files at handoff time.
- `results/processed/`: summaries, coverage, selectors, validity results, and runtime
  reports.
- `results/runs/`: additive per-generation JSON logs with environment, Git metadata,
  timestamps, status, and SHA-256 output hashes.
- Several early Phase 0–2 CSVs remain directly under `results/` for historical
  filename compatibility.

Do not overwrite raw data with processed data. Do not silently drop failed,
ill-conditioned, unresolved, nonphysical, or not-applicable rows.

Important large artifacts:

- `results/raw/qaoa_selection_v1.csv`: 4,500 candidate-level rows, approximately
  46 MiB.
- `results/raw/calibration_lifetime_v1.csv`: 60,480 rows, 25 columns, approximately
  11 MiB.
- Readout scalability/mismatch raw artifacts account for most repository data size.

### `figures/`

Generated PNGs for all research tracks. Phase G adds:

- `figures/calibration_age_vs_decision_error.png`
- `figures/amortized_cost_vs_reliability.png`

Both were visually inspected after final generation. The first shows error versus
calibration age by drift model; the second uses four workload panels to avoid label
overlap and show amortized cost versus decision error.

### `docs/`

Scientific and reproducibility contracts:

- `definitions.md`: terminology, evidence labels, seed and fixed-budget policy,
  deterministic readout solve.
- `limitations.md`: evidence boundaries.
- `extended_experiment_contract.md`: preregistration, oracle isolation, statistics,
  fixed-budget and hardware boundaries.
- `extended_research_questions.md`: questions and falsifiers.
- `zne_validity_protocol.md`: frozen ZNE acceptance/rejection logic.
- `readout_scalability_protocol.md`: Phase E protocol and resource gates.
- `qaoa_selection_protocol.md`: full Phase F application contract.
- `selector_rules.md`: frozen non-ML selector.
- `benchmark_schema.md` and `adaptive_benchmark_schema.md`: result schemas.
- `day08_12_runtime_report.pdf`: measured/estimated runtime report for an earlier batch.
- `releases/v0.1.0.md`: older simulator release candidate notes; verify freshness
  before treating this as a current release description.

### `paper/`

- `paper/main.tex`: two-column technical paper source.
- `paper/main.pdf`: compiled seven-page PDF, 1,162,054 bytes at handoff time.
- `paper/references.bib`: numeric `abbrv` bibliography with 10 cited entries.
- `paper/generated/`: CSV-derived LaTeX macros and table fragments.
- `paper/figures/`: paper-local copies of authoritative figures.

The paper currently includes work through Phase F. It does not include Phase G.

### `outputs/`, `tmp/`, and caches

- `tmp/` and `outputs/` are ignored and should not be treated as authoritative.
- `.pytest-tmp/`, `.ruff_cache/`, `.coverage`, and `.matplotlib-cache/` are local
  state and ignored.
- Do not commit virtual environments, caches, LaTeX auxiliaries, logs, or `.env` files.

## 3. Environment & setup

### Supported Python

- Supported: Python 3.11 or Python 3.12.
- CI uses Python 3.12.
- The verified local environment used Python 3.12.14.
- `pyproject.toml` rejects Python 3.13 and later (`>=3.11,<3.13`).

### Fresh Windows/PowerShell setup

From a normal clone:

```powershell
Set-Location "C:\path\to\adaptive-qem-benchmark"

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-ci.txt
```

Activation is optional. Calling the venv Python directly is the least ambiguous
Windows workflow:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If activation is desired:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python --version
```

The process-scoped execution-policy change ends with the PowerShell process. Do not
change the machine or user execution policy merely to activate this venv.

### Dependency sources

Preferred reproducible installation:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-ci.txt
```

Pinned versions at handoff time:

- matplotlib 3.10.7
- coverage 7.11.0
- numpy 2.3.4
- pandas 2.3.3
- pypdf 6.21.4
- pytest 8.4.2
- qiskit 2.2.3
- qiskit-aer 0.17.2
- ruff 0.14.3
- scipy 1.16.3

Editable/project metadata installation is optional. If needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Do not mix unpinned upgrades into a reproducibility investigation. First reproduce
with `requirements-ci.txt`.

### Current local-worktree environment caveat

The handoff worktree itself does **not** contain `.venv`. During development it used
the parent environment:

`C:\Users\wits\Documents\Codex\2026-10-30\adaptive-qem-benchmark\.venv`

This is an implementation-machine detail, not a repository requirement. A new clone
should normally create its own `.venv` at the clone root. Do not delete or recreate
the existing parent `.venv` unless the user explicitly requests it; it contains the
working pinned tools.

### Deterministic environment variables

Before artifact generation or exact CI reproduction:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = ".matplotlib-cache"
```

The BLAS variables reduce cross-platform numerical variation. `Agg` avoids Tk GUI
failures during figure generation.

### PowerShell path behavior

- PowerShell does not execute a program in the current directory without `.\`.
  Use `.\.venv\Scripts\python.exe`, not `.venv\Scripts\python.exe`.
- Quote every path containing spaces.
- Prefer `-LiteralPath` for file operations.
- Avoid mixing PowerShell path enumeration with `cmd.exe` deletion commands.

### Git dubious-ownership workaround

Codex-created worktrees may be owned by a sandbox Windows account rather than the
interactive user. Do **not** add `safe.directory=*` and do not modify global Git
configuration. Use a process-only setting for the exact clone:

```powershell
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "safe.directory"
$env:GIT_CONFIG_VALUE_0 = "C:/Users/wits/Documents/Codex/2026-10-30/adaptive-qem-benchmark/work/phase-d-publish"
git status -sb
```

If `.git/index.lock` is denied from an AI sandbox, the interactive Windows user may
need to perform `git add`, `git commit`, and `git push`. Do not change ACLs or take
ownership without explicit user approval.

### LaTeX/PDF setup

The paper is compiled with Tectonic. Tectonic is not declared as a Python dependency
and may not be globally installed. The development machine had an external executable
at:

`C:\Users\wits\Documents\Codex\2026-10-30\adaptive-qem-benchmark\work\tectonic-0.17.0\tectonic.exe`

That path is outside the repository and is not portable. TODO: verify or install a
trusted Tectonic binary before the next paper rebuild.

From the repository root, when `tectonic` is on `PATH`:

```powershell
tectonic paper/main.tex --outdir paper
```

Equivalent from `paper/`:

```powershell
tectonic main.tex --outdir .
```

Tectonic warnings about underfull boxes have occurred. The last Phase F PDF was
visually inspected page by page and had no clipping, overlap, blank pages, or missing
figures. Do not claim a later PDF is valid without recompiling, checking page count
and required text, rendering every page, and visually inspecting it.

## 4. Tooling & commands

All Python tools below should be installed inside the venv. Git and Tectonic are
external executables.

### Fast status check

```powershell
git status -sb
git log -5 --oneline
```

### Ruff

CI scope:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
```

Full local repository scope used during recent phases:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Do not run automatic formatting or broad `--fix` without reviewing the diff. The
repository follows a surgical-change policy.

### Pytest

Full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The last full local Phase G verification passed. The output represented 277 tests at
that point. TODO: verify the current collected count with:

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

Targeted Phase G:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_calibration_drift.py -q
```

Targeted Phase F:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qaoa_selection.py tests/test_qaoa_selection_runner.py tests/test_qaoa_selection_artifacts.py -q
```

### Coverage

```powershell
.\.venv\Scripts\python.exe -m coverage erase
.\.venv\Scripts\python.exe -m coverage run -m pytest
.\.venv\Scripts\python.exe -m coverage report
```

The configured minimum is 50% branch-aware project coverage. The latest Phase G run
reported 61% total coverage and 75% for `src/calibration_drift.py`.

### Git diff integrity

```powershell
git diff --check
git status -sb
git diff --stat
git diff --name-only
```

### Phase G generation

Direct deterministic generation:

```powershell
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = ".matplotlib-cache"
.\.venv\Scripts\python.exe -m src.calibration_drift
```

With additive run logging:

```powershell
.\.venv\Scripts\python.exe -m src.run_with_log src.calibration_drift
```

The generator takes only a few seconds on the development machine. It writes:

- `results/raw/calibration_lifetime_v1.csv`
- `figures/calibration_age_vs_decision_error.png`
- `figures/amortized_cost_vs_reliability.png`

### Per-run logging

General syntax:

```powershell
.\.venv\Scripts\python.exe -m src.run_with_log <module.path> [module arguments]
```

Logs are written under `results/runs/` and include UUID, timestamps, wall time,
Python/package versions, Git state, host, arguments, root seed, output SHA-256 hashes,
status, and error text. They do not embed result data.

### Artifact comparator

Help:

```powershell
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --help
```

Modes:

```powershell
# Compare current working-tree artifacts against HEAD without regeneration.
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --current

# Phase F-only current-tree comparison; no expensive QAOA rerun.
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --phase-f

# Regenerate the normal generator set in an isolated clone and compare it.
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --run-log-output results/runs

# Also run expensive scalability, mismatch, and QAOA generators.
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --include-expensive
```

The comparator uses `rtol=1e-9` and `atol=1e-12` for numeric CSV columns. It checks
row counts, columns, nonnumeric values, missing/extra artifacts, and structural PDF
requirements where configured. PNGs are checked for existence/nonzero size rather
than byte identity because rendering bytes can differ by platform.

**Do not casually run the full comparator.** The normal CI generator set includes
`src.generate_nontrivial_benchmark`, which has taken roughly 40 minutes. The
`--include-expensive` mode adds:

- `src.generate_readout_scalability_benchmark`
- `src.generate_readout_mismatch_benchmark`
- `src.generate_qaoa_selection`

Those can add substantial runtime and regenerate large artifacts.

The normal generator set currently includes:

- `src.generate_artifacts`
- `src.regenerate_readout_affected`
- `src.generate_nontrivial_benchmark`
- `src.generate_adaptive_analysis`
- `src.generate_zne_probability_diagnostics`
- `src.generate_zne_falsification`
- `src.generate_zne_validity_gate`
- `src.readout_scalability_metrics`
- `src.generate_readout_tensor_validation`
- `src.generate_readout_subspace_validation`
- `src.generate_readout_matrix_free_validation`
- `src.calibration_drift`
- `src.generate_audit_manifest`

Note that `src.generate_combined_qem` is not currently in the comparator's generator
tuple; combined-QEM committed artifacts are protected primarily by their tests.
Changing this behavior requires a deliberate runtime/reproducibility decision.

### CI-equivalent checks

GitHub Actions uses:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = ".matplotlib-cache"

.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m coverage run -m pytest
.\.venv\Scripts\python.exe -m coverage report
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --run-log-output results/runs
```

CI runs on `ubuntu-latest` with Python 3.12 and installs
`requirements-ci.txt` using `--only-binary=:all:`.

### Paper validation

After compiling:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from pypdf import PdfReader; p=Path('paper/main.pdf'); print(p.exists(), p.stat().st_size, len(PdfReader(p).pages))"
```

Also check unresolved references/text extraction and visually render every page.
Poppler was available in the Codex bundled runtime during development, but a new
assistant must locate its own `pdftoppm` or use an equivalent trusted renderer.

## 5. Work completed so far

### Foundational decisions

1. **Fixed total budget, not free per-method overhead**
   - Calibration states, folded circuits, candidate circuits, and all other
     executions are charged to the declared total budget.
   - This prevents readout or ZNE from receiving hidden extra information.

2. **Deterministic seeds**
   - Every stochastic seed derives from `ROOT_SEED = 20261102` through
     `src.config.derive_seed()`.
   - Python's randomized `hash()` is never used for scientific seeds.

3. **Preserve evidence levels**
   - `analytic`
   - `noiseless_simulated`
   - `noisy_simulated`
   - `hardware_derived`
   - Some extended contracts also allow `inferred_hypothesis`.
   - No current result may be labeled hardware-derived.

4. **Failures are data**
   - Explicit statuses are required.
   - Failed, ill-conditioned, nonphysical, unresolved, abstained, and
     not-applicable rows are retained.

5. **No ML and no hardware in current scope**
   - The selector is deterministic and rule-based.
   - Hardware execution needs a separate protocol and explicit authorization.

### Core and volumetric findings

As documented in `README.md` and the paper:

- 144 GHZ/hardware-efficient operating cells produced:
  - 103 practical ties;
  - 18 readout winners;
  - 14 ZNE winners;
  - 9 inconclusive cells.
- The frozen rule selector achieved 41.4% decision compatibility on a 29-cell
  deterministic holdout and 50.0% under four unseen synthetic noise shifts.
- 179 of 432 method summaries had negative RMSE gain relative to raw estimation.
- 99 explicit hardware-efficient readout failures were retained.

These findings support regime dependence and abstention as research questions, not a
claim that the selector is generally reliable.

### ZNE validity and combined-QEM decisions

- Apparent movement toward an exact reference is not accepted as credible ZNE by
  itself.
- The validity gate checks signal retention, physicality, paired gain uncertainty,
  trend, and advantage over matched-cost garbage folding.
- Only 5 of 16 tested cells passed the frozen ZNE validity gate; none of the tested
  hardware-efficient cells passed every check.
- Combined QEM had the lowest expectation RMSE in 12 of 16 preregistered conditions,
  but all 16 combined-ZNE conditions failed at least one validity criterion.
- At a 0.02 decision margin, combined-QEM decision-error rates ranged approximately
  from 0.26 to 0.48 and did not consistently beat readout mitigation.

The reasoning decision was to separate estimator improvement from mechanism validity
and from downstream decision reliability.

### Scalable readout findings

- Full assignment calibration grows exponentially with width.
- Tensor and matrix-free methods completed all 1,440 rows each in the unified
  scalability study.
- Full matrix correction failed 38 width-8 repeats as ill-conditioned and was
  resource-limited at widths 10/12.
- At width 8, mean full-matrix solve time was approximately 10.9 seconds versus
  approximately 0.057 seconds for matrix-free correction.
- Every method amplified sampling standard deviation on average.
- Under deliberate calibration mismatch, a representative width-6 broad-output,
  16,384-shot cell had decision-error probability approximately 0.88–0.94 across
  correction methods.

The interpretation is deliberately narrow: lower classical memory does not imply
lower shot cost, lower uncertainty, or robustness to a wrong calibration model.

### Phase F: exact MaxCut/QAOA application

Primary files:

- `src/applications/qaoa_selection.py`
- `src/generate_qaoa_selection.py`
- `experiments/configs/qaoa_selection_v1.yaml`
- `experiments/manifests/qaoa_candidate_pool_v1.csv`
- `experiments/manifests/qaoa_selection_v1_manifest.csv`
- `results/raw/qaoa_selection_v1.csv`
- `results/processed/qaoa_selection_summary_v1.csv`
- `docs/qaoa_selection_protocol.md`
- three `qaoa_*.png` figures

Design decisions:

- exact classical MaxCut uses exhaustive enumeration and preserves degenerate
  optimal bitstrings;
- candidate generation uses graph structure and noiseless statevector values only;
- noisy/mitigated outcomes cannot create, remove, or reorder candidates;
- one computational-basis shot yields one complete cut sample—edges do not receive
  independent free shots;
- selection ties use a deterministic preregistered rule while preserving all tied
  IDs;
- certification uses nested checkpoints and Bonferroni simultaneous coverage;
- the runtime selector never receives the exact ideal winner.

Headline Phase F findings:

- 4,500 candidate-level raw rows and 900 selection groups.
- 4,140 rows completed; 360 candidate rows failed, corresponding to 72 failed
  medium/dense readout selection groups.
- All completed clear-ranking decisions selected the ideal best candidate.
- Near-tie top-1 error:
  - raw: 0.28 at 4,096 shots; 0.17 at 16,384;
  - readout: 0.32 among completed 4,096-shot selections; approximately 0.179 at
    16,384;
  - ZNE: 0.36 at 4,096; 0.42 at 16,384.
- Only 2 of 828 completed selection groups certified a winner; 826 completed groups
  remained unresolved.
- Lower energy RMSE did not reliably improve top-1 selection.
- Exact top-1 accuracy and simple regret led to different emphasis because a wrong
  near-tie choice can have low regret.

### Phase G: calibration amortization and drift

Primary files:

- `src/calibration_drift.py`
- `tests/test_calibration_drift.py`
- `experiments/configs/calibration_lifetime_v1.yaml`
- `results/raw/calibration_lifetime_v1.csv`
- `figures/calibration_age_vs_decision_error.png`
- `figures/amortized_cost_vs_reliability.png`

Frozen scientific contract:

- true one-state probability: 0.53;
- decision threshold: 0.50;
- target shots per decision: 1,024;
- calibration shots per event: 1,024;
- initial readout probabilities: `p01=0.03`, `p10=0.05`;
- fixed interval: 10 decisions;
- uncertainty trigger:
  `0.002 + calibration_age * 0.0025 >= 0.012`;
- 30 repeats;
- assignment convention: `A[observed_state, prepared_state]`.

The directed slow, abrupt, and asymmetric models move the corrected decision toward
the 0.5 boundary; `random_bounded` is a bidirectional control. This choice is explicit
in the config because early prototyping showed that drift directed away from the
boundary can make stale calibration look beneficial and fail to test the intended
decision-risk mechanism.

Policies may use only decision index and calibration age. They may not inspect the
true drift matrix or exact decision outcome. Policy comparisons use common target and
calibration random numbers where schedules coincide, preventing seed noise from
creating artificial policy differences.

Artifact facts:

- 60,480 rows;
- 25 columns;
- 60,480 unique target seeds;
- all rows status `complete`;
- CSV and both PNGs reproduced with bitwise-identical SHA-256 hashes across two
  consecutive local generations.

Average results across the four drift models:

| Workload | Policy | Shots/decision | Total workflow shots | Decision error |
|---:|---|---:|---:|---:|
| 1 | all policies | 2,048.00 | 2,048 | 0.050 |
| 5 | every decision | 2,048.00 | 10,240 | 0.038 |
| 5 | fixed/trigger/never | 1,228.80 | 6,144 | 0.035 |
| 20 | every decision | 2,048.00 | 40,960 | 0.062 |
| 20 | fixed interval | 1,126.40 | 22,528 | 0.073 |
| 20 | uncertainty triggered | 1,228.80 | 24,576 | 0.061 |
| 20 | never | 1,075.20 | 21,504 | 0.109 |
| 100 | every decision | 2,048.00 | 204,800 | 0.062 |
| 100 | fixed interval | 1,126.40 | 112,640 | 0.074 |
| 100 | uncertainty triggered | 1,228.80 | 122,880 | 0.067 |
| 100 | never | 1,034.24 | 103,424 | 0.234 |

At workload 100, never-recalibrate error by drift model was approximately:

- abrupt change: 0.386;
- asymmetric: 0.331;
- slow monotonic: 0.170;
- random bounded control: 0.047.

Interpretation: calibration reuse can substantially reduce cost, but indefinite reuse
is unsafe under directed drift. The uncertainty-triggered policy approached
every-decision reliability for directed drift while using about 40% fewer shots per
decision. This is synthetic one-qubit evidence only, not a hardware calibration
lifetime claim.

### Paper and bibliography

The paper uses a two-column, high-information-density `article` layout with sections:

- Abstract
- Introduction
- Methodology
- Experiment Setup
- Results
- Conclusion
- Discussion
- Reproducibility appendix
- References

The bibliography uses numeric `abbrv` style. Existing sources include foundational
QEM/ZNE/readout papers, Qiskit Aer documentation, scalable readout mitigation,
benchmarking, artefactual ZNE improvement, and decision-risk/sampling-cost work.
Do not add citations not actually read or supported.

## 6. Open tasks / next steps

### Priority 1 — verify the remote state and CI

Do this before editing:

```powershell
git status -sb
git log -3 --oneline
git fetch origin
git status -sb
gh run list --limit 5
```

Expected latest implementation commit: `1f2c572`.

If Git reports divergence, stop and report it. Do not force-push or reset. If the
latest CI run failed, inspect the exact failed step and logs before changing code.

### Priority 2 — integrate Phase G into public documentation

Update `README.md` with:

- the calibration-reuse research question;
- exact workload/drift/policy definitions;
- per-decision and workflow cost accounting;
- the workload-100 headline comparison;
- the random-bounded control result;
- explicit one-qubit/synthetic/no-hardware limitations;
- a reproduction command for `src.calibration_drift`.

Do not alter Phase G results or regenerate unrelated artifacts merely to improve the
narrative.

Recommended new documentation:

- `docs/calibration_lifetime_protocol.md`

It should freeze the matrix convention, drift equations, trigger rule, common-random-
number pairing, cost contract, and evidence boundary. Derive it from the config and
implementation; do not invent new parameters after seeing outcomes.

### Priority 3 — integrate Phase G into the paper

Update `paper/main.tex` and paper-local assets:

1. Add methodology for assignment-matrix age, deterministic drift, amortized cost,
   and recalibration policies.
2. Add experiment setup: one-qubit threshold task, 30 repeats, workload lengths,
   shots, and policy information boundary.
3. Add results using both Phase G figures.
4. Discuss the key tradeoff:
   - every-decision calibration is most expensive;
   - triggered calibration retained similar directed-drift reliability at lower cost;
   - never recalibrating failed under directed drift but not under the random bounded
     control;
   - conclusions are sensitive to drift direction and workload length.
5. Explicitly state that physical plausibility and hardware calibration lifetime are
   not established.
6. Copy Phase G figures into `paper/figures/` if the paper remains self-contained.
7. Recompile `paper/main.pdf`.
8. Parse PDF structure, confirm page count and required strings, then render and
   visually inspect every page.

Do not claim the Phase G model is calibrated to IBM or any physical backend. Do not
add a hardware citation solely to make the simulation appear more realistic.

### Priority 4 — consider a processed Phase G summary

The committed Phase G deliverable contains raw rows and plotting-time summaries but
no processed CSV. For auditability, consider adding:

`results/processed/calibration_lifetime_summary_v1.csv`

The summary should be generated, not hand-written, and should contain at least:

- workload length;
- drift model;
- policy;
- decisions and complete decisions;
- decision-error probability, preferably with a Wilson interval;
- mean calibration age;
- calibration events;
- amortized shots per decision;
- total workflow shots.

If added, update tests and comparator coverage. Do not overwrite the raw CSV.

### Priority 5 — close the Phase G reproducibility gate

After documentation/paper changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_calibration_drift.py -q
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m coverage run -m pytest
.\.venv\Scripts\python.exe -m coverage report
git diff --check
```

If code or Phase G artifacts changed, rerun only:

```powershell
.\.venv\Scripts\python.exe -m src.calibration_drift
```

Do not run Phase A–F generators unless their authoritative inputs changed.

The current comparator has no Phase G-only CLI flag. Options:

- compare the known Phase G files with repeated `--only-artifact` plus `--current`;
- or add a minimal `--phase-g` scoped set analogous to `PHASE_F_ARTIFACTS`.

If adding `--phase-g`, keep the change small and add regression tests. Do not run the
expensive QAOA or scalable-readout generators merely to validate Phase G.

### Priority 6 — user-directed next research phase

No Phase H specification is currently frozen in the repository. Do not invent and
execute a new study without the user's direction. Plausible extensions should remain
ideas until explicitly approved, especially hardware work, richer drift models, and
multi-qubit calibration lifetime studies.

### Commit/push convention

The user prefers a commit after each meaningful task slice and exact PowerShell push
commands. Do not silently skip Git failures. Because AI-sandbox Git writes can be
blocked, it is acceptable to finish verified files and give the user exact commands.

Suggested commit for this handoff only:

```powershell
git add HANDOFF.md
git commit -m "Add project handoff documentation"
git push origin main
```

Use a separate commit for Phase G README/paper integration.

## 7. Known issues & caveats

### Scientific limitations

- Simulator-only; no hardware data.
- Synthetic noise and drift models do not model full device drift, leakage,
  relaxation, non-Markovian noise, or calibrated crosstalk.
- The main GHZ/hardware-efficient benchmark is limited to width 6.
- Scalable readout tests reach width 12, but resource limits and synthetic models
  constrain interpretation.
- The QAOA study has only three deliberately controlled graph instances, five
  ideal-only candidates per graph, and `p=1`.
- Phase G is a one-qubit threshold decision with directed stress models. Its purpose
  is mechanism isolation, not realistic hardware lifetime prediction.
- Candidate or drift parameters must not be tuned because results are unfavorable.

### Reproducibility/CI fragility

- Historical CI failures were caused by stale generated artifacts, platform-specific
  last-bit matrix-inversion differences, PNG rendering differences, and audit/PDF
  byte-comparison assumptions.
- Current CSV comparisons are numeric-tolerant (`rtol=1e-9`, `atol=1e-12`). Do not
  widen tolerances to hide real differences.
- PNGs are structurally checked for nonempty output, not byte equality.
- Runtime PDFs are structurally checked rather than byte-compared.
- Readout correction uses a deterministic small-system algorithm to avoid BLAS pivot
  instability. Do not replace it casually with platform-dependent inversion.
- CI can take around 40 minutes because the normal comparator includes nontrivial
  benchmark regeneration.
- `--include-expensive` can be much slower and should not be used routinely.

### Git and filesystem issues

- Sandbox-created clones may trigger Git dubious-ownership errors.
- `.git/index.lock` can be unwritable to an AI process even when project files are
  writable.
- Do not use `safe.directory=*`.
- Do not change global/system Git configuration unless the user explicitly asks.
- Do not use `git reset --hard`, force-push, or destructive cleanup.
- Preserve unrelated working-tree changes.
- If a temporary pytest directory is access-denied, do not recursively change ACLs or
  delete it without approval. Pytest already uses `.pytest-tmp` as its base temp.

### Environment issues

- Do not delete or recreate the user's parent `.venv` without explicit approval.
- Always set `MPLBACKEND=Agg` for headless plot generation; a previous Tk backend
  caused a plot-stage failure after expensive computation.
- Pin BLAS threads to one for artifact reproduction.
- Tectonic is external and not guaranteed to be on `PATH`.
- The repository contains large CSVs. Avoid loading every raw file simultaneously.

### Data and artifact rules

- Never rewrite committed raw observations as corrected or aggregated values.
- Never silently drop failed rows.
- Never clip out-of-range ZNE estimates merely to make them physical.
- Preserve factor-level ZNE and readout calibration records where the schema requires
  them.
- Do not rename existing files without explicit instruction; CI and paper references
  depend on stable paths.
- Do not edit any external Obsidian planning file. It is documentation outside the
  build and has repeatedly been declared out of scope.
- Run logs are additive. Do not embed raw datasets inside run-log JSON.

### Paper caveats

- `paper/main.pdf` is currently seven pages and valid through Phase F.
- Phase G integration will likely change page count; do not preserve seven pages as an
  arbitrary target.
- PDF compilation success, structural validation, citation validation, and visual
  inspection are separate gates.
- Do not add unsupported citations or claims.

### README/documentation freshness

- `README.md` does not yet mention Phase G.
- Some older limitations paragraphs describe intermediate Phase E states. Read the
  full section and reconcile stale statements before publication.
- `docs/releases/v0.1.0.md` predates several later phases. Treat it as historical
  unless explicitly updated.

## 8. Key context for continuity

### Mandatory conventions

1. **Think before coding**: surface assumptions and tradeoffs.
2. **Simplicity first**: implement only the requested scope.
3. **Surgical changes**: every changed line should trace to the task.
4. **Goal-driven execution**: define measurable success and verify it.

These principles are repeated in the user's `AGENTS.md` instructions. Follow the
repository's existing style rather than refactoring adjacent code.

### Naming and evidence conventions

- Configs/manifests/results generally use a version suffix such as `_v1`.
- Parameters should appear in filenames or explicit row columns.
- Use `noisy_simulated`, not “experimental” or “hardware,” for finite-shot Aer and
  classical sampling results.
- Use “observed” only for findings actually supported by generated data.
- Label explanations as hypotheses or possible causes unless experimentally isolated.
- Keep null, negative, failed, and unresolved results visible.

### Statistical conventions

- Fixed total shot budgets across methods.
- Repeat-level raw rows preserved.
- Wilson intervals for decision-error probabilities.
- Bias, variance, RMSE, decision error, unresolved probability, and overhead reported
  separately.
- Equal-budget method comparison does not imply equal numbers of circuit executions.
- Common random numbers are used only when scientifically valid and must not grant a
  method uncharged information.

### Paper conventions

- Two-column `article` layout.
- Default order: Abstract, Introduction, Methodology, Experiment Setup, Results,
  Conclusion, Discussion, References.
- Numeric `abbrv` bibliography.
- Captions should state conclusions, not merely describe axes.
- Clearly distinguish analytic, simulated, experimental, and hardware-derived data.
- Move secondary derivations to appendices.

### External resources

- GitHub repository: <https://github.com/Raina-ya/adaptive-qem-benchmark>
- Qiskit Aer documentation: <https://qiskit.github.io/qiskit-aer/>
- The bibliography already contains the relevant arXiv references and metadata. Use
  `paper/references.bib` as the source of truth rather than reconstructing citations
  from memory.

Previously supplied research references represented in `paper/references.bib`:

- arXiv:2108.12518 — scalable measurement-error mitigation;
- arXiv:2107.13470 — QEM benchmarking;
- arXiv:2607.09360 — artefactual improvements in ZNE;
- arXiv:2607.02888 — accuracy, decision risk, and sampling cost in QEM.

Do not assume the PDFs themselves are tracked in the repository; they are not listed
as project files. Use the bibliography metadata and obtain papers from authoritative
sources if deeper citation verification is required.

### Current authoritative checkpoints

- Latest implementation commit: `1f2c572`.
- Phase F integration commit: `e24cab6`.
- Root seed: `20261102`.
- Python: 3.11/3.12; CI 3.12.
- Local verified full coverage: 61%, configured minimum 50%.
- Phase G raw rows: 60,480, all complete.
- Phase F candidate rows: 4,500; 72 failed selection groups; 2 certified completed
  groups.
- Paper: seven pages before Phase G integration.

### First five actions for Claude

1. Read `AGENTS.md`, this handoff, `README.md`, and
   `experiments/configs/calibration_lifetime_v1.yaml`.
2. Run `git status -sb`, inspect the latest commit, and verify current remote/CI state.
3. Run targeted Phase G tests and Ruff before editing.
4. Integrate Phase G into README and paper without changing existing Phase G data.
5. Rebuild and visually inspect the PDF, run full verification, then commit/push as a
   distinct task slice.

