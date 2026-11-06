# Release Checklist

This checklist rebuilds results from a clean clone, without reusing committed outputs. The scope
tool deletes each declared output first, so a generator cannot load an old result. A rebuild supports
reproducibility only. It does not show that the simulator models a device.

## Tiers

| Tier | When it runs | Checks |
| --- | --- | --- |
| Routine CI | Every push | Tests, schema, hashes, coverage, cross-file consistency, and cheap regeneration |
| Scoped audit | On demand | One clean rebuild of a named scope, compared within tolerance |
| Release audit | Before a release | Clean rebuild of every generated artifact, then tests and the paper |

Declared inputs that are not rebuilt are listed in `experiments/configs/artifact_registry_v1.yaml`:
frozen inputs, measured runtime data, QAOA pilot and smoke files, and unreferenced legacy files.

## Scopes

| Scope | Rebuilds | Time on the development desktop |
| --- | --- | --- |
| `combined_decision` | Combined-QEM and decision benchmarks | About 2 hours |
| `expensive` | Scalable readout, mismatch, QAOA, and selector v2 | About 2 hours |
| `release` | All of the above and every cheaper generator | About 5 hours |

Other machines will differ. A GitHub job stops at 350 minutes, so a slow `release` run may need a
local machine.

## Run on GitHub

```powershell
gh workflow run release-audit.yml -f scope=combined_decision
gh run list --workflow release-audit.yml --limit 3
```

The workflow uses a fresh checkout and locked dependencies. It rebuilds and compares, runs the tests,
builds the paper, validates it, and uploads the run logs and PDF.

## Run locally

```powershell
git clone https://github.com/Raina-ya/adaptive-qem-benchmark "$env:TEMP\qem-clean"
Set-Location "$env:TEMP\qem-clean"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --scope combined_decision --dry-run
.\.venv\Scripts\python.exe -m src.verify_generated_artifacts --scope combined_decision --workdir "$env:TEMP\qem-regenerated" --run-log-output "$env:TEMP\qem-logs"
Set-Location "$env:TEMP\qem-regenerated"
..\qem-clean\.venv\Scripts\python.exe -m pytest --deselect tests/test_artifact_provenance.py::test_committed_hashes_match_the_tree
```

Build the paper in `paper/` with `pdflatex`, `bibtex`, then `pdflatex` twice, and run
`python -m src.verify_final_project`.

## Reading the result

- **Pass:** every declared output matches within relative tolerance 1e-9 and absolute tolerance 1e-12.
  PNG files are only checked to be nonempty.
- **Fail:** keep the committed results and investigate. Do not loosen a tolerance or replace results
  to make the audit pass. A difference between platforms is a finding to record.
- **After an intended change:** run `python -m src.artifact_provenance --write` and review the diff.

## Before a release

- [ ] Routine CI is green on the release commit.
- [ ] The `release` scope passes, or each difference is documented.
- [ ] `python -m src.artifact_provenance` reports no failures.
- [ ] `python -m src.claim_ledger` reports no mismatches.
- [ ] The rebuilt PDF has the expected page count and required text.
- [ ] Run logs are attached to the release.
