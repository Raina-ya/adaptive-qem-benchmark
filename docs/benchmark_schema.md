# Volumetric Benchmark Schema

`volumetric_v1` is a simulator-only, repository-precommitted GHZ benchmark. Each manifest row is
one `(width, depth, noise, shots, method, repeat)` cell. GHZ has one preparation-depth
parameter, so depths 3 and 6 remain visible and carry `not_applicable` rather than being
silently removed.

| Column | Meaning |
|---|---|
| width | Number of qubits |
| depth | Declared depth parameter |
| noise | Mixed simulated-noise level |
| shots | Total fixed shot budget, including mitigation overhead |
| method | `raw`, `readout`, or repository-precommitted linear `zne` |
| repeat | Zero-based repeat ID |
| status | `pending`, `complete`, `failed`, or `not_applicable` |
| seed | Deterministic child seed derived from the root seed |

Raw repeat-level observations are immutable inputs to processed coverage and plots. ZNE
splits shots across factors 1, 3, and 5. Readout mitigation charges calibration shots to
the same total budget. All values are simulated; the reference is analytic.
