# Independent Cross-Validation

`src/independent_reference.py` recalculates core results by enumeration. It imports no project code
and no Qiskit. `src/cross_validation.py` compares it with committed results and records the outcome in
`results/processed/cross_validation_v1.csv`. Deliberate mistakes are in
`results/processed/cross_validation_mutations_v1.csv`. With real Qiskit installed,
`tests/test_cross_validation.py` also compares the reference with live statevectors.

## Conventions compared

| Convention | Statement |
| --- | --- |
| Bit order | Basis integer bit i is qubit i. Measured strings have qubit 0 rightmost, and Pauli strings list qubit 0 first. |
| Normalization | Counts are divided by their own total, and corrected distributions are scaled to sum to one. |
| Calibration matrix | Rows are observed states and columns are prepared states. |
| Shot accounting | Remainders go to the earliest parts, and calibration uses a fifth of the method's shots. |
| Ranking and ties | The highest estimate wins, and ties within 1e-12 are resolved by candidate identifier. |
| Degeneracy | A selection is correct if it matches any exact winner. |
| QAOA phase | Each edge applies `RZZ(-gamma)`, followed by `RX(2 beta)` on every qubit. |

## Results

- **Exact values:** Bell, GHZ, nine hardware-efficient circuits, and fifteen QAOA candidates agree within
  1e-12. Stored hardware-efficient references have 12 significant digits, which limits that comparison.
- **Fixed counts:** Across all completed QAOA selections, including readout at width 8, selected
  candidates, rankings, top-k recall, and shot accounting match exactly. Energies and regret also match
  within tolerance, apart from the limitation below.
- **Readout cell:** All 50 repeats of `he_w2_d1` under asymmetric readout match, including the matrix,
  condition number, corrected distribution, and expectation.

Tolerances are 1e-12 for exact values and 1e-9 for calculations that use different solvers, including
the condition number. A difference of 1e-11 was seen there.

## Findings

- **Width-8 readout energies:** These are not exact. The first path's iterative solver stops at a
  relative step of 1e-9, so its energies differ from the certified optimum by up to about 9e-7 (median
  about 1e-8). No selection or ranking changes among the 28 completed groups. This row stays marked as
  not passing, with a stated limitation. The tolerance was not widened.
- **SciPy NNLS:** The first draft of the reference used SciPy alone, which returned a non-optimal point
  on a committed input (`sparse6`, repeat 47) and made the comparison look like a first-path error of up
  to 8e-4. Checking optimality showed the first path was correct, so the reference now certifies the
  KKT conditions and rejects any answer that fails them.
- **Symmetric cells:** A bit reversal changes nothing for Bell, GHZ, full-support Paulis such as `XZ`,
  `ring4`, or `sparse6`, because each maps onto itself. Passing there says nothing about bit order, so the
  mutations also use `ZI` and `medium_dense8`.

## Limits

- Noisy sampling is not cross-checked. Counts are inputs here, so this does not verify Aer's noise channels,
  folding, or the sampled distributions.
- Only selected cells are covered. Most `estimate` values across the benchmark are still not rebuilt from
  counts.
- The same author wrote both paths after reading the first, so a shared misunderstanding remains possible.
  Both also rely on NumPy for its random angles and linear algebra.
