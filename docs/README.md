# Documentation Index

The repository is complete as a simulator study. This index separates the canonical
scientific documentation from historical build records. Numerical phase or day names
that remain in artifact paths record the original execution order; they are not the
current organization of the research.

## Start here

- `methodology.md`: experiment design, evidence labels, and statistical conventions.
- `limitations.md`: simulator, model, statistical, and generalization boundaries.
- `cross_validation.md`: independent enumerated checks of exact values and selected outputs.
- `release_checklist.md`: clean-clone reproduction and release verification.
- `release_rebuild_v1.md`: measured Windows/Linux regeneration differences.
- `hardware_readiness_review.md`: documented no-go decision for hardware execution.

## Decision reliability

- `accuracy_vs_decision_risk.md`: separation between estimation accuracy and decision risk.
- `qaoa_mechanism.md`: common-mode and contrast-error analysis for candidate selection.
- `qaoa_selection_protocol.md`: fixed-budget MaxCut/QAOA protocol.
- `sequential_decision_protocol.md`: sequential certification and repeated-look control.
- `confirmatory_selector_certification_protocol.md`: confirmatory power, selector, and
  certification protocol.
- `confirmatory_operational_boundary_results.md`: corresponding confirmatory and
  operational-boundary results.

## Mitigation validity and operating boundaries

- `zne_validity_protocol.md`: ZNE validity criteria and evidence boundary.
- `readout_scalability_protocol.md`: full, tensor, subspace, and matrix-free correction.
- `readout_mismatch_drift_protocol.md`: mismatch dose response, feasibility, and drift policy.
- `signal_v2_protocol.md` and `signal_v2_results.md`: signal-stratified holdout study.
- `selector_audit_protocol.md` and `selector_rules.md`: selector definitions and no-go audit.

## Schemas and contracts

- `definitions.md` and `extended_definitions.md`: terminology and conventions.
- `benchmark_schema.md` and `adaptive_benchmark_schema.md`: result schemas.
- `extended_experiment_contract.md`: extended benchmark contract.
- `observable_floor_contract.md`: observable-floor controls.

## Historical records

- `archive/foundation_benchmark_completion.md`: original foundation acceptance checklist.
- `day08_12_runtime_report.pdf`: historical runtime audit for the first ZNE and volumetric batch.
- `releases/v0.1.0.md`: earlier release note retained for provenance.

The obsolete daily planning checklist and daily progress note were removed from the
canonical tree. Their history remains available through Git.
