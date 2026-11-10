# Results and Artifact Status

All committed results are simulator, analytic, or derived artifacts as labeled in their
schemas. No file in this directory is hardware-derived.

## Layout

- `raw/`: immutable repeat-level observations, counts, statuses, and charged resources.
- `processed/`: deterministic summaries, coverage tables, audits, and claim provenance.
- `runs/`: compact generator metadata and output hashes; no full result data are embedded.
- `day02_...` through `day09_...`: immutable foundation artifacts from the original build
  sequence.

The `dayXX` prefix is historical. These files remain at their original paths because tests,
generators, hashes, manifests, and the evidence ledger treat those paths as reproducibility
contracts. A day number does not imply that the completed project is still organized as a
daily task plan.

## Immutability and supersession

Do not rename, overwrite, or silently correct a committed raw artifact. Corrections and new
analyses receive a new versioned artifact, while the earlier file remains available for audit.
Known superseded outputs are identified in the README, documentation, or artifact registry;
for example, the original regime-label file is retained because its labeling defect and the
corrected replacement are part of the public evidence trail.

`experiments/configs/artifact_registry_v1.yaml` is the machine-readable index for generated
artifacts, generators, hashes, and row counts. Use `python -m src.artifact_provenance` to
validate the committed tree without rerunning expensive simulations.
