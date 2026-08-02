# Estimator Contracts

Status: frozen for the SA0/SA1 audit packet on 2026-08-02.

The audit maintains two estimator contracts. Their artifacts, configuration
names, and result directories must remain separate.

## `legacy_reproduction`

Purpose: reconstruct the historical analysis exactly enough to identify the
effect of each discrepancy in the published record.

This contract preserves the executed behavior of the pinned historical code,
including:

- the realized simulation feature-space correlation;
- the executable Monte Carlo replication count;
- contiguous deterministic fold construction;
- independently standardized fit, validation, and test partitions where that
  occurred historically;
- fixed three-repeat response averaging with unacquired repeats represented as
  zeros where that occurred historically;
- the historical lambda grid, solver, and score semantics; and
- the historical output tuple and map calculations.

Outputs from this contract are evidence about reproducibility. They are not
the default estimator for new scientific work.

## `corrected_v2`

Purpose: define the CPU float64 oracle for future scientific analyses and GPU
parity.

This contract requires:

1. Finite two-dimensional feature and target arrays with explicit row and
   column identities.
2. Immutable, externally supplied inner and outer splits.
3. An explicit acquisition-availability mask; response averages use acquired
   repeats only, and images with no acquired response are excluded.
4. No overlap between fit and validation rows.
5. No overlap between fit and validation groups when groups encode repeated
   images, sessions, participants, or another protected unit.
6. Every preprocessing transform fitted on the fit rows and applied unchanged
   to validation and test rows.
7. Predictions and out-of-fold errors returned in raw target units.
8. Per-target ridge regularization selected only inside the applicable
   training partition.
9. Signed held-out coefficient of determination with an explicitly tested
   constant-target convention.
10. Nonnegative stacking coefficients that sum to one, accompanied by
   feasibility, objective, and convergence diagnostics.
11. Saved intermediate identities for transforms, selected regularization,
    out-of-fold predictions, errors, mixture optimization, test predictions,
    and scores.

Every corrected run declares one feature-transform policy. The initial CPU
oracle uses train-partition-fitted transforms. After GPU float64 parity passes,
`none_full_feature` may retain the original feature coordinates and use a
reviewed primal/dual ridge implementation. It is a separate scientific arm and
cannot be substituted for a historical PCA run. See
[`GPU_FULL_FEATURE_ARM.md`](GPU_FULL_FEATURE_ARM.md).

The implementation-independent mathematical and artifact specification is
frozen in [`CORRECTED_V2_SPEC.md`](CORRECTED_V2_SPEC.md). The first deterministic
CPU float64 fixture and its intermediate hashes are recorded in
[`CORRECTED_V2_FIXTURE_RESULT.json`](CORRECTED_V2_FIXTURE_RESULT.json).

Mixture coefficients are predictive weights. With identical or duplicated
experts, individual coefficients are nonidentifiable; equivalence is assessed
using ensemble prediction, objective value, and aggregate weight over the
equivalent experts.

## Boundary between contracts

- A run declares exactly one contract before reading outcomes.
- A `legacy_reproduction` output cannot be relabeled `corrected_v2`.
- A correction comparison may place outputs side by side but may not merge
  their intermediate arrays.
- GPU float32 is an additional numerical mode, not a third scientific
  contract. It is blocked until `corrected_v2` float64 parity passes.
