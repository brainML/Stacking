# SA2 Simulation Provenance Audit

Date: 2026-08-02

Status: source-level discrepancy localized; outcome replay not yet run.

## Published correlation discrepancy

Immediately before public commit `92cb5cd`, the active definitions accepted a
`correl` argument but did not use it when sampling feature spaces or targets.
Consequently, setting `correl=0.2` generated the same distribution as
`correl=0`, apart from ordinary random-number state and any metadata label.

Commit `92cb5cd` changed the generator so that, for positive `correl`, each
feature space receives a projected component from the next feature space and
the target receives a corresponding added component. The final feature space
and its target contribution are scaled separately.

The corrected parameter should therefore be described as a **correlated-
component mixing amplitude**, not as a guaranteed realized Pearson
correlation of exactly 0.1, 0.2, or 0.3. The replay must report empirical
within- and between-space covariance/correlation diagnostics for every setting.

## Extended-sweep count discrepancy

The written correction states:

- 576 parameter settings; and
- 100 independent datasets per setting.

The executable configuration defines 50 runs per setting. Therefore it builds:

```text
4 sample sizes x 3 dimension patterns x 3 alpha patterns
x 4 noise values x 4 mixing amplitudes x 50 runs
= 28,800 simulation tasks
```

Each task has two output targets, so a completed result table can contain
57,600 rows. A row count of 57,600 must not be interpreted as 100 independently
generated datasets per setting.

## Random-number pairing

The sweep seeds each task from a global index. Changing a parameter, including
the mixing amplitude, also changes the seed. This is valid for estimating
independent Monte Carlo averages, but it prevents paired attribution of a
difference to correlation alone and increases comparison variance.

The corrected replay will use both:

1. independent Monte Carlo replicates for distributional summaries; and
2. common-random-number pairs across mixing amplitudes for direct impact
   attribution.

The pairing scheme and seed derivation must be frozen before results are read.

## Required next evidence

1. Recover the exact notebook/script and random state that produced every
   published simulation figure.
2. Inventory the untracked CORTEX sweep arrays and figures without modifying
   the dirty worktrees.
3. Compute realized feature correlation/canonical-correlation summaries rather
   than reporting only the mixing parameter.
4. Run the historical generator and corrected generator with matched latent
   draws on a small frozen grid.
5. Reconcile 50 versus 100 independent datasets before any full sweep.
6. Separate estimator effects from generator effects by evaluating legacy and
   corrected estimators on the same generated datasets.

No conclusion about stacking versus concatenation is released at this stage.
