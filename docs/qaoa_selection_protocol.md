# QAOA Candidate-Selection Application Protocol

## Scope

This track is a simulator-only MaxCut/QAOA application case study. It does not use
machine learning, quantum hardware, hardware calibration data, or noisy outcomes to
construct application instances. Phase F-A defines only exact references and ideal
circuit primitives; it does not execute the QEM benchmark.

## Frozen graph foundations

The application starts with three deterministic simple graphs:

- `ring4`: a four-vertex cycle;
- `sparse6`: a connected six-vertex graph whose additional edges are selected by
  root-derived deterministic edge scores;
- `medium_dense8`: a connected eight-vertex graph with a deterministic medium-density
  edge set.

Edges are immutable, unique, sorted, and stored as `(lower_vertex, upper_vertex)`.
The graph seed is derived from root seed `20261102`; Python's randomized `hash()` is
never used.

## Independent classical reference

For a computational-basis state, one shot yields one complete bitstring. Its cut
value is the number of graph edges whose endpoint bits differ. All edge contributions
come from that same shot; edges do not receive independent or uncharged shots.

Exact MaxCut enumerates only states with the highest-index qubit fixed to zero, which
removes the global complement symmetry. After finding the optimum, the implementation
adds each complementary state back and returns every degenerate optimum. The exact
classical optimum is distinct from every QAOA expectation value.

## Qiskit ordering and QAOA convention

Basis integers use Qiskit's little-endian qubit convention. Displayed and sampled
bitstrings are `q[n-1]...q[0]`, so the rightmost character is `q0`.

The QAOA cost function is

`C = sum_(i,j in E) (I - Z_i Z_j) / 2`.

Each layer applies `rzz(-gamma)` on every canonical edge and `rx(2*beta)` on every
qubit after an initial uniform superposition. The interface accepts equal-length
beta and gamma sequences, although Phase F-A validates the `p=1` foundation.

## Evidence labels

- `analytic`: exhaustive classical MaxCut optimum and optimal bitstrings;
- `noiseless_simulated`: exact statevector QAOA probabilities and expected cut;
- `noisy_simulated`: reserved for later finite-shot simulator runs;
- `hardware_derived`: absent from this project.

Expected cut from a probability vector is an exact statevector-derived quantity.
Expected cut from counts is a finite-shot estimator. These values must not be
described interchangeably.

## Frozen candidate construction

Phase F-B freezes QAOA depth at `p=1`. Depth two is disabled until the noiseless
candidate study and the later `p=1` noisy benchmark are reviewed. The complete pool
is the Cartesian grid

- `beta_i = pi*i/24`, for `i=0,...,12`;
- `gamma_j = pi*j/12`, for `j=0,...,23`.

This creates 312 candidates per graph. Every pool row, including candidates not
selected for the noisy benchmark, is preserved in
`experiments/manifests/qaoa_candidate_pool_v1.csv`. Candidates are ranked by exact
noiseless statevector expected cut, descending, with candidate ID as the deterministic
tie breaker. No noisy, mitigated, or hardware-derived value enters pool generation,
ranking, or subset selection.

Known limitation: candidates with mathematically equal energies differ in their last
floating-point digits, so the committed order within those ties came from the Windows
machine that generated the pool, not from the candidate-ID tie breaker. A fresh Linux
computation orders them differently, which could change the selected candidates. The
committed pool and manifest are therefore frozen inputs: selection reads the committed
ranking, and generation does not replace an existing pool. Tests check that fresh
energies match within `1e-9` and that committed ranks remain ordered by energy. The tied
choice used only noiseless values, so no noisy outcome influenced it, but the selected
candidates cannot be derived from scratch on another platform.

The benchmark requests five candidates per graph: the incumbent, the closest
nonzero-gap candidate within 2% of the exact classical optimum, the first challenger
at least 10% of the exact optimum below the incumbent, a median-rank control, and a
lowest-rank control. If the near-tie rule has no solution, it is reported rather than
relaxed. `ring4` has no nonzero-gap candidate within the frozen 2% threshold and uses
an additional non-tied control; `sparse6` is explicitly a control graph. The selected
manifest preserves candidate roles, exact ideal energies, ranks, and gaps.

## Selection budget

For one graph, method, and repeat, the total application budget is `B`, shared by all
candidates and every method-specific execution. The frozen budgets are 4,096 and
16,384 shots. Exactly 50% is reserved for candidate selection and 50% for final
evaluation. Later readout calibration, ZNE factors, and folded executions must fit
inside the same `B`; they cannot be added as uncharged overhead. The Phase F-B
manifest records this contract but does not execute it.

Phase F-C freezes the simulator condition at `mixed_medium`. Within the selection
half of `B`, raw sampling divides all shots across the five candidates. Readout
mitigation spends 20% of the selection half on one assignment-matrix calibration
shared by the five candidates, then divides the remainder across their target
circuits. Linear ZNE divides the selection half first across candidates and then
across preregistered fold factors 1, 3, and 5. The untouched final-evaluation half
remains reserved; it is not silently reused during selection.

Raw and readout target circuits use a common-random-number seed derived without the
method label. This pairs simulator randomness where the physical target circuit is
the same, but differing shot counts remain fully charged. ZNE factor seeds are
independent deterministic children of the same candidate seed because folded circuits
are distinct executions. Calibration data are shared only within one graph, budget,
repeat, and readout-method group; no method receives another method's samples.

Every output row retains one candidate even after execution failure. Ill-conditioned
readout correction, solver failure, and ZNE values outside the physical cut range are
reported explicitly. ZNE estimates are never clipped. Candidate execution counts and
selection-group execution counts are separate, and a complete bitstring shot supplies
all edge contributions without creating extra per-edge executions.

The frozen reporting value is `top_k = 2`. The pilot uses three repeats; the intended
full run uses 50. Changing the allocation, candidate set, or `k` after inspecting
noisy outcomes would create a new protocol version.

## Application decision metrics and certification

The runtime selector sees only finite-shot estimates. It never receives exact ideal
energies. Estimates are sorted from largest to smallest; an exact numerical tie within
`1e-12` is broken by lexicographically smallest candidate ID, while every tied ID is
retained in the output. Exact ideal degeneracy is treated as a set: selection of any
exact maximizer is correct, and the unique-winner indicator is `not_applicable`.

For each repeat, the benchmark reports set-valued top-1 correctness, top-2 recall,
simple regret `max_i E_i - E_selected`, and energy RMSE across all candidates. Top-2
recall uses the set of all exact candidates at or above the second-place energy, so an
exact tie at the boundary is not silently broken. Across repeats, best-candidate
selection probability and decision-error probability use resolved repeats only;
unresolved probability is reported separately. Decision-error and unresolved rates
receive two-sided 95% Wilson intervals. Simple regret and exact top-1 accuracy remain
separate because a wrong near-tie decision can have negligible regret.

Certification uses cumulative nested checkpoints at 25%, 50%, 75%, and 100% of the
selection-shot allocation. Samples at a later checkpoint must extend the earlier
sample prefix; they are not independent reruns. Let `m` be the number of candidates,
`T=4` the number of sequential looks, and `alpha=0.05`. Every candidate/checkpoint
interval uses

`z* = Phi^-1(1 - alpha / (2*m*T))`,

`L_i(t) = estimate_i(t) - z* SE_i(t)`, and
`U_i(t) = estimate_i(t) + z* SE_i(t)`.

The current noisy winner `w` is certified at the first checkpoint satisfying
`L_w(t) > max_(j != w) U_j(t)`. This Bonferroni union bound provides at least 95%
simultaneous coverage over candidates and sequential looks without assuming that
candidate estimates or checkpoints are independent. A noisy tie cannot be certified.
If separation never occurs, status remains `unresolved`, shots-to-decision is missing,
and the maximum charged budget is retained.

Shot overhead counts readout calibration and, for ZNE, factor-3 and factor-5 samples
relative to the factor-1 estimate. Execution overhead is the selection-group circuit
count divided by the raw one-execution-per-candidate baseline. These quantities do not
change the fixed total budget.

For full assignment matrices below dimension 128, correction retains the fixed-order
active-set NNLS solver. At dimensions 128 and above, the same nonnegative least-squares
objective uses deterministic projected FISTA with a fixed tolerance and iteration cap.
A Varah diagonal-dominance bound replaces explicit dense inversion for the large-matrix
condition gate. A non-dominant matrix is conservatively reported as ill-conditioned;
non-convergence is retained as a failed row. This solver switch changes classical
runtime, not calibration data, shot allocation, or the correction objective.

Checkpoint standard errors use the empirical complete-cut variance for raw sampling.
Readout checkpoint errors multiply that raw standard error by the assignment-matrix
condition estimate, a deliberately conservative amplification bound. ZNE propagates
the three independent factor-level empirical variances through the preregistered
linear-intercept weights. These approximations are used only for stopping/certification;
they do not alter the reported point estimates.

## Sampled-cut verification

Every sampled computational-basis bitstring has a cut value that can be evaluated
exactly by classical computation. For these small graphs, the preregistered exact
success indicator is `best sampled cut == exact MaxCut optimum`. Complementary
bitstrings represent the same cut and require no special quantum-side treatment.
The additional approximate metric is the fraction of samples whose cut is at least
`alpha * optimum`, with `alpha = 0.95`.

These are externally verified application metrics. A quantum sampler cannot prove
global optimality by itself. The word *certified* in this track means that the
small-n classical exhaustive solver independently confirmed the sampled value; it
does not mean quantum-verifiable optimality and does not scale once exact classical
solution becomes intractable.

## Current limitations

The instances have at most eight vertices and remain classically trivial. They test
accounting, ranking, and mitigation behavior rather than quantum advantage. Candidate
parameters, ranking, `top_k`, the 50/50 selection/final allocation, noise model,
method-specific budget splitting, application metrics, and certification rule are now
frozen. The readout condition-amplified interval is conservative rather than a full
bootstrap of constrained correction, and this limitation must accompany claims about
the low certification rate.
