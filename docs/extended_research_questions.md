# Extended Research Questions

## Purpose

The extension asks whether quantum error mitigation improves a downstream decision
under honest resource accounting, not merely whether it changes an expectation-value
estimate. All primary questions are falsifiable and all current evidence remains
analytic or simulator-derived.

## Primary questions and falsifiers

### Decision reliability

**Question:** Can a QEM method lower expectation RMSE while increasing downstream
sign, argmax, or ranking error?

**Falsifier:** Across preregistered margins, circuits, noise conditions, budgets, and
paired repeats, no method exhibits lower expectation RMSE and higher decision-error
probability with uncertainty intervals that resolve the ordering.

**Question:** Which exact decision margins remain unresolved under a fixed total shot
budget?

**Falsifier:** Every preregistered margin is resolved under every applicable method and
budget according to the frozen uncertainty rule.

### ZNE validity

**Question:** Can linear ZNE report apparent improvement after amplified circuits have
lost useful signal relative to the observable floor?

**Falsifier:** Every apparent improvement occurs only in cells that retain signal at
the amplified factors, remain physically bounded, and outperform a matched-cost
negative control.

### Combined mitigation

**Question:** Do readout mitigation and ZNE correct distinguishable error sources, or
does their combination merely compound variance and execution overhead?

**Falsifier:** The preregistered combined method never produces a resolved interaction
or practical gain beyond the better single method after charging all calibration and
folded-circuit shots.

**Question:** Is any combined-method gain large enough to justify its compounded
overhead?

**Falsifier:** Every combined result is dominated in the declared cost-and-accuracy
comparison, practically tied, failed, or inconclusive.

### Readout scalability

**Question:** Can readout mitigation preserve useful accuracy and uncertainty bounds
without constructing the full `2^w` assignment matrix?

**Falsifier:** Reduced or matrix-free correction fails the frozen small-width
agreement tests or becomes inaccurate, unstable, or resource-dominated as width and
observed support increase.

### Hardware readiness

**Question:** Would optional hardware execution discriminate between explicit
hypotheses, or would it only decorate a simulator study?

**Falsifier for readiness:** No proposed hardware cell has frozen simulator
predictions, sufficient metadata, a declared budget, and outcomes that would
distinguish competing explanations. In that case the correct result is `no_go`.

## Non-claims

- Simulator performance is not evidence of performance on a quantum device.
- A lower scalar error is not automatically a more reliable decision.
- An extrapolated value closer to a reference is not automatically valid ZNE.
- A selector is not useful merely because it matches an oracle label inside a
  practical tie.
- More elaborate mitigation is not assumed to be better.

## Evidence progression

The required order is analytic construction, noiseless statevector verification,
noisy finite-shot simulation, frozen holdout evaluation, and only then an optional
hardware go/no-go review. Hardware execution is not authorized by this contract.
