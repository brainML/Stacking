# Corrected V2 CPU Oracle Specification

Date frozen: 2026-08-02

Status: **implementation-independent SA3 specification**.

## Inputs and identities

Each run supplies finite float64-compatible two-dimensional feature matrices
`X_f` and a target matrix `Y` with identical rows, unique row IDs, unique
feature-space IDs, and unique target IDs. One immutable outer split and at
least two immutable inner splits are supplied as explicit row indices. The
outer split partitions every row. Within the outer-fit rows, every inner split
is a complete fit/validation partition and the validation sets form exactly
one pass. Protected groups cannot cross any fit/held-out boundary.

Repeated responses are represented as `repeat x image x target` plus an
explicit acquisition mask. Only acquired, finite observations enter each
average. Images with zero acquired repeats are excluded before splits are
formed.

## Feature transforms

Every transform is fitted on the applicable fit rows and applied unchanged to
validation or test rows.

- `standardize`: subtract the fit-column mean and divide by the fit-column
  population standard deviation. Constant columns receive scale one.
- `nested_pca`: first apply the same standardizer, then compute an exact
  float64 SVD on fit rows and retain a prespecified component count. Component
  signs are fixed by making the largest-magnitude loading nonnegative.

`nested_pca` is the initial reduced-feature scientific policy. The
`none_full_feature` arm remains blocked until CPU/GPU float64 parity and is not
another name for the small-fixture `standardize` test mode.

Targets are standardized using fit-row statistics only during ridge fitting.
Every validation, OOF, and test prediction is inverse-transformed and returned
in raw target units.

## Nested ridge

For expert `f`, target `v`, and positive candidate `lambda`, solve

\[
\hat B_{f,v,\lambda}
= (X_f^\top X_f + \lambda I)^{-1}X_f^\top Y_v.
\]

The mathematically equivalent dual form is used when transformed feature
dimension exceeds fit-row count. Candidate lambdas are selected separately
per target by minimum total inner-validation mean squared error in raw target
units. Ties resolve to the first lambda in the prospectively supplied grid.
The selected lambda is then used to produce complete inner OOF predictions and
one final outer-test prediction from an outer-fit transform and refit.

## Predictive mixture

Let `E_v` be the `expert x outer-fit-row` matrix of raw-unit OOF residuals for
target `v`. Define

\[
P_v = E_v E_v^\top / n_{fit}.
\]

The CPU oracle solves

\[
\min_w w^\top P_v w
\quad\text{subject to}\quad w \ge 0,\; \mathbf{1}^\top w=1.
\]

For at most 12 experts, every nonempty active set is examined in deterministic
order. Feasible equality-constrained candidates are compared by objective.
The result records objective value, sum error, minimum weight, active-set
stationarity, inactive dual violation, complementarity, and active count.

Individual coefficients are predictive mixture weights, not unique feature
importance. For duplicated or identical experts, validation concerns ensemble
prediction, objective, and aggregate equivalence-class weight.

## Scores and required intermediates

Outer-test performance is per-target held-out signed R2. Constant targets have
score zero. A complete oracle result retains or hashes:

- row, target, feature, and split identities;
- inner and final transform identities;
- per-target lambda losses and selected lambdas;
- raw-unit expert OOF and test predictions;
- final ridge weights and resolved primal/dual form;
- residual quadratic matrices through reproducible inputs;
- simplex weights and all diagnostics;
- raw-unit ensemble OOF and test predictions; and
- expert and ensemble signed R2.

The reference fixture must be bitwise deterministic within one environment,
pass primal/dual prediction equivalence at `1e-10`, and pass simplex feasibility
and KKT diagnostics at their frozen float64 tolerances before GPU parity work.
