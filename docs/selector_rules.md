# Frozen Selector v1

The selector is a deterministic rule policy. It does not use machine learning and never
reads an exact reference, estimation error, oracle label, or benchmark winner at
inference time. Inputs are declared noise condition, shot budget, width, depth,
transpiled gate count, calibration conditioning, and ZNE fit residual.

Rules are evaluated in this fixed order:

1. Abstain when the calibration condition number exceeds 8.
2. Abstain when the ZNE residual exceeds 0.03.
3. Use raw estimation for a declared zero-noise condition.
4. Use raw estimation for width 6 or greater at a 1,024-shot budget.
5. Use readout mitigation for low, medium, or asymmetric-readout noise when calibration
   conditioning is at most 3.
6. Use ZNE for deep high, coherent, or correlated gate-noise conditions when its fit
   residual is at most 0.01.
7. Return inconclusive for a circuit with more than 100 counted gates and fewer than
   4,096 shots.
8. Otherwise use raw estimation because no mitigation rule has sufficient evidence.

`abstain` and `inconclusive` are first-class outcomes. Rule traces are stored with every
prediction. The policy is a frozen repository rule. Its rules and first holdout results appear in the same
commit, so Git history cannot show that the rules came first.
