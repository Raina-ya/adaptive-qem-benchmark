# Why Lower Energy RMSE Can Mean Worse Selection

**Status: exploratory.** This reanalyzes committed repeat rows and runs no circuits. `medium_dense8` was
chosen after its counterexample was seen. The other two graphs were also seen before this analysis, so
they test consistency and are not a holdout. Everything is conditional on three graphs, `p=1`, five frozen
candidates, one synthetic noise model, and 50 repeats per cell. Run `python -m src.qaoa_mechanism` to rebuild
the `results/processed/qaoa_mechanism_*_v1.csv` tables and `figures/qaoa_mechanism_*.png`.

## Definitions

For graph g, repeat r, candidate c, and method m:

- error: `e = estimate - ideal energy`
- common part: `b(g,r,m)`, the mean of `e` over candidates in that repeat
- contrast part: `d(g,r,c,m) = e - b`

These give `energy MSE = mean(b^2) + mean(d^2)` exactly, and a test enforces it. Adding a constant to all
candidates cannot change which one has the highest estimate, so `b` cannot affect selection, by construction.

## Answers

### What does ZNE improve?

It improves contrast error, not common bias in any consistent way. Contrast RMSE falls in all six ZNE cells,
with bootstrap intervals that exclude zero. Common bias RMS gets worse in three cells, improves in two, and is
unresolved in one. Common bias is also small: 0.09 of the raw 0.81 in the dense 16,384-shot cell. Energy RMSE
is therefore almost entirely contrast error, and the split into "common" and "contrast" does not by itself
explain the selection loss.

### Why can RMSE 0.66 lose to RMSE 0.81?

These are the dense graph at 16,384 shots, where top-1 error is 0.50 for ZNE and 0.18 for raw.

1. **Most of the RMSE gain is on a candidate that never contends.** MSE falls from 0.659 to 0.435. The
   low-energy control provides 77% of that drop, because its error shrinks from +1.64 to +1.35. Its ideal
   energy is 4.72, far below the leader's 10.22, so it never decides anything.
2. **Selection depends on the gap between the top two candidates.** The ideal gap is 0.168.

   | Method | Mean estimated gap | Gap SD | Mean / SD | Predicted inversion | Observed inversion |
   | --- | --- | --- | --- | --- | --- |
   | Raw | 0.045 | 0.053 | 0.85 | 0.20 | 0.18 |
   | ZNE | 0.016 | 0.137 | 0.12 | 0.45 | 0.50 |

3. **ZNE raises the spread of that gap.** Linear fit weights for folds 1, 3, and 5 are 13/12, 1/3, and -5/12.
   With shots split evenly among folds, the estimator's variance grows by 4.375, so its SD grows by 2.09. The
   observed ratio across the six cells is 1.92 to 2.58. This is a resource and variance effect.

The leaders' own RMSE also improves, from 0.53 to 0.39, because their large errors are mostly shared within
the pair. An accuracy gain can therefore be real even on the contenders and still not help the choice.

Changing one factor at a time in the normal approximation moves predicted inversion from 0.20 to 0.37 with
ZNE's spread alone, and to 0.38 with ZNE's mean alone. The mean's share is uncertain: the gap fell from 0.045
to 0.016, but the standard errors are 0.007 and 0.019, so this drop is not clearly resolved.

### Is this specific to `medium_dense8`?

| Feature | Dense | Sparse | Ring | Verdict |
| --- | --- | --- | --- | --- |
| ZNE lowers energy and contrast RMSE | Yes | Yes | Yes | Recurs |
| ZNE roughly doubles the gap SD | 2.05, 2.58 | 1.94, 2.51 | 2.02, 1.92 | Recurs |
| Non-contenders supply most of the MSE | 83-86% | 42-51% | 42-47% | **Dense only** |
| Top-1 selection point estimate worsens at 16,384 shots | -0.32 | -0.18 | -0.08 | Direction recurs |
| Survives Holm correction | Only 16,384 shots | No | No | **One cell only** |

Across all 11 mitigation comparisons, top-1 accuracy never rises. Its point estimate falls in 9 and ties in 2. After Holm correction, only dense ZNE at 16,384 shots is resolved (22 raw-only versus 6 ZNE-only correct,
adjusted p = 0.041). At 4,096 shots on the sparse graph, ZNE and raw tie (0.28 each). The failure to reach
significance elsewhere may reflect 50 repeats rather than absence of an effect.

The dominance of the low-energy candidate is instance-specific. The doubling of the gap's spread follows from
the fit and recurs in all cells, so it is the better candidate for a general mechanism.

## A testable consequence

Under the normal approximation and equal shots, ZNE helps a two-candidate decision only if it raises the mean
gap by more than its SD factor of about 2.09. Removing all bias can raise the raw gap at most to the ideal gap:

| Graph | Ideal gap / raw gap at 16,384 shots | Can ZNE break even? |
| --- | --- | --- |
| Ring | 1.33 | No |
| Sparse | 1.32 | No |
| Dense | 3.73 | Possibly, but the observed gap ratio is 0.36 |

The sparse graph at 4,096 shots is the one ZNE cell with a ratio near break-even (1.90), and it is the one ZNE
cell that ties. That ratio exceeds the 1.36 bias removal could give at that budget, so sampling noise contributes.
This is a hypothesis generated from the same data and needs a test on new graphs.

## Rejected idea

The working hypothesis was that raw errors are highly correlated across candidates, which would keep rankings
stable. The data do not support this. Candidates are sampled independently, so mean off-diagonal error
correlation across repeats is near zero for raw and ZNE, with 9% of the 170 correlations beyond the 0.28 expected
to appear by chance at about 5%. Readout is higher in some cells (up to 0.21), which may reflect its shared
calibration. Raw keeps its order because noise shrinks the gap only moderately while the spread stays small.

## Limits

- This is one simulated noise model, and no device data exists.
- The normal approximation is a description here, though it tracks observed inversions closely.
- ZNE and raw share repeat indices, but ZNE draws different samples, so pairing gains little precision.
- The QAOA raw file has not yet had a clean-clone rebuild (the `expensive` scope). A failed rebuild could change
  these numbers.
- Readout at width 8 has only 28 completed repeats at 16,384 shots and none at 4,096.
- Paper text is unchanged for now; it is planned for the paper-focused part of this work.
