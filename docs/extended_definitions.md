# Extended Definitions

## Decision objects

- **Candidate:** One circuit-observable pair whose reference value participates in a
  downstream comparison.
- **Exact decision margin:** For two candidates A and B,
  `reference_A - reference_B`, computed analytically or by noiseless statevector
  simulation before finite-shot sampling.
- **Estimated decision margin:** `estimate_A - estimate_B` for one repeat and method.
- **Sign decision:** A choice determined by the sign of one reference or estimated
  scalar. A zero estimate follows the documented tie rule rather than being
  silently counted as correct.
- **Argmax decision:** Selection of the candidate with the largest estimated value.
  Candidate ordering and tie breaking are fixed before sampling.
- **Top-k decision:** Selection of an ordered or unordered set of `k` candidates;
  the exact convention must be declared in the experiment configuration.
- **Decision error:** Indicator that the sampled decision differs from the exact
  decision. Exact estimated ties count according to the frozen tie policy.
- **Simple regret:** Exact value of the optimal candidate minus the exact value of
  the selected candidate. It is zero for an optimal selection and non-negative for a
  maximization task.
- **Unresolved decision:** A result whose frozen confidence or uncertainty rule does
  not certify one side of the decision boundary. It is not forced into a correct or
  incorrect label.

## Estimation and dependence

- **Gap error:** Estimated decision margin minus exact decision margin.
- **Candidate covariance:** Repeat-level covariance between paired candidate
  estimates. It enters gap variance as
  `Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)`.
- **Shared records:** Candidate estimates that intentionally reuse a declared source
  of randomness, calibration, or measurement records. The shared component must be
  identified explicitly.
- **Independent records:** Candidate estimates generated from disjoint stochastic
  records after deterministic seed derivation.

## Operating regimes

- **Practical tie:** Two or more methods differ by less than the documented
  practical-effect threshold after cost accounting. A tie is not a unique winner.
- **Abstention:** A selector output that declines to recommend a mitigation method
  because no method lies inside the frozen reliability conditions.
- **Decision compatibility:** A selected method is compatible with an oracle outcome
  when it is the unique oracle choice or belongs to the documented equivalent set
  for a practical tie.
- **Cost-aware utility:** A documented objective combining scientific error or
  decision loss with explicitly normalized execution overhead. Its formula and
  penalty cannot change after holdout inspection.

## ZNE validity

- **Observable floor:** The limiting expectation or probability approached after
  useful signal is destroyed under the declared noise process. It is
  observable-specific and is not assumed to be zero.
- **Signal retention:** The amount of reference-to-floor contrast remaining at an
  amplified noise factor, normalized by the corresponding scale-one contrast when
  that denominator is identifiable.
- **Extrapolation artefact:** An apparent improvement attributable to extrapolation
  arithmetic, model violation, signal destruction, or an unphysical reconstruction
  rather than credible recovery of the target signal.
- **Garbage-folding control:** A matched-cost negative control that increases signal
  destruction without representing the intended unitary noise-scaling mechanism. It
  tests whether the extrapolation pipeline can manufacture apparent gain.
- **Validated ZNE gain:** A gain that passes the frozen signal-retention, physical
  bound, uncertainty, and negative-control requirements. Apparent gain is reported
  separately.

## Resource and deployment terms

- **Total shot budget:** All target, calibration, folded, training, selection, and
  other auxiliary shots charged to one declared comparison.
- **Execution overhead:** Target and auxiliary circuit executions relative to the
  declared raw baseline. Shot overhead and circuit-execution overhead are recorded
  separately.
- **Calibration amortization:** Allocation of one calibration cost across a declared
  number of downstream decisions. Both total-workflow and per-decision costs remain
  visible.
- **Hardware readiness:** A go/no-go state based on frozen simulator predictions, a
  falsifiable device-level question, a declared execution budget, complete metadata,
  and adequate controls. `no_go` is a valid scientific conclusion.

## Evidence labels

- `analytic`: derived without numerical circuit execution.
- `noiseless_simulated`: computed by exact or effectively exact noiseless simulation.
- `noisy_simulated`: sampled from a recorded simulator and noise configuration.
- `hardware_derived`: produced by an identified physical-device job with required
  metadata.
- `inferred_hypothesis`: a proposed explanation not directly established by the
  reported evidence.

The term **observed** is reserved for a supported result. Possible explanations are
labelled hypotheses and cannot be promoted to observations without a discriminating
test.
