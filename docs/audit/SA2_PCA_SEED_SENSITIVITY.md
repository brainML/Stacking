# SA2 Response-Free PCA Seed Sensitivity

Date: 2026-08-02

Status: **completed; historical PCA is not seed-invariant**.

No recoverable seed, PCA object, RNG state, or seeded job wrapper was found for
the participant-2 structured-variance scripts. The recovered wrappers invoke
`python stack_vp.py` or `python stack_vp2.py` without setting a seed. Nearby
banded-ridge and sphere-ridge scripts use 1337, but that does not establish the
state of a separate Python process.

## First diagnostic

The first computation is response-free and uses only participant-2 pooled
AlexNet conv3 features:

- full matrix shape `(10000, 9600)`;
- historical `PCA(n_components=512, svd_solver="auto")` behavior;
- prospectively frozen seeds 0, 1337, and 2026;
- 256 deterministic metadata-spanning probe rows; and
- no neural response loading, ridge fitting, stacking, or historical-output
  comparison.

Seed 1337 is included because it appears in adjacent historical scripts, not
because it is presumed correct. Seed 2026 identifies the present audit, and 0
is a conventional neutral control. Results cannot be used to select whichever
seed best supports a scientific conclusion.

The primary comparisons are invariant to component sign and within-subspace
rotation:

- principal-angle cosines between component subspaces;
- subspace chordal distance;
- relative Frobenius difference between probe-row Gram matrices; and
- total explained-variance ratio.

Raw component and probe-transform hashes are recorded only for deterministic
rerun identity.

## Result

All three runs resolved to scikit-learn's randomized PCA solver. Total
explained-variance ratios were nearly identical (range `1.21e-5`), but that
aggregate hides differences in the retained subspaces:

- mean principal-angle cosines ranged from 0.9659 to 0.9666;
- the least-aligned direction in each pair had cosine 0.0012 to 0.0435;
- chordal distances ranged from 5.0456 to 5.0628; and
- relative probe-row Gram differences ranged from 2.408% to 2.438%.

Therefore a fresh randomized PCA draw cannot be treated as an exact,
seed-invariant reconstruction of the historical preprocessing. This result
does not by itself establish that held-out encoding scores or mixture weights
change materially; that is the next bounded gate.

The sanitized metrics and provenance hashes are recorded in
[`SA2_PCA_SEED_SENSITIVITY_RESULT.json`](SA2_PCA_SEED_SENSITIVITY_RESULT.json).
An initial artifact whose unbounded SVD roundoff produced cosine maxima a few
parts per million above one was superseded. The validated result clips only
the computed cosine spectrum to its mathematical `[0, 1]` range; PCA outputs,
probe-Gram comparisons, and component hashes are unchanged.

## Escalation rule

The first branch failed: the retained feature geometry is not effectively
invariant. The next run will fit all seven experts for the same three seeds but
restrict downstream evaluation to one outer fold and 256 prospectively fixed
voxels. Do not search a large seed space for the seed that best matches the
published outcomes.

That downstream gate is prospectively frozen in
`configs/audit/legacy_seed_downstream_subject02.json`. Voxel indices are chosen
by evenly spanning `[0, 5000)` without reading responses or historical outputs.
The primary gate tests held-out prediction correlations and stacked signed-R2
differences. Weight L1 differences and top-expert agreement are reported as
interpretive-instability flags, not used to rescue or reject predictive
stability. Full response-derived arrays remain private on MIND.

## Downstream result

The bounded downstream gate completed all three seed fits. Its batch process
then exited while assembling the summary because a local variable shadowed
Python's package-metadata module. All three immutable seed artifacts were
already present; the validated summary was reconstructed from them without a
model refit. The runner is corrected and a dedicated recovery finalizer is
versioned.

Every seed pair failed the prospectively frozen predictive gate and raised the
interpretive-instability flag:

- median voxelwise prediction correlation: 0.9410--0.9424;
- fifth-percentile prediction correlation: 0.5971--0.6399;
- median absolute stacked signed-R2 difference: 0.00049--0.00054;
- 95th-percentile signed-R2 difference: 0.00423--0.00589;
- median mixture-weight L1 difference: 0.2298--0.2977; and
- top-weight expert agreement: 72.3%--78.1%.

Thus aggregate accuracy is relatively stable while voxelwise predictions and
the purportedly interpretable mixture weights are not. A fresh PCA draw cannot
serve as an exact historical reconstruction, and choosing the seed closest to
the published map would be post hoc. Published artifacts remain the sole
`published_historical` tier; any new seeded execution must be labeled
`legacy_reconstruction`.

Sanitized aggregate results and provenance hashes are in
[`SA2_SEED_DOWNSTREAM_RESULT.json`](SA2_SEED_DOWNSTREAM_RESULT.json). The next
method-development gate is the deterministic `corrected_v2` CPU float64 oracle,
with seed uncertainty retained as an explicit sensitivity analysis.

The diagnostic is configured in
`configs/audit/pca_seed_sensitivity_subject02_conv3.json`. It requests one
Slurm CPU job, 16 CPUs, 64 GiB RAM, four hours, and no concurrency. The bounded
feature-only job completed on 2026-08-02 in the recovered environment.
Scheduler identifiers, cluster paths, and logs remain in the ignored private
run manifest; the public record contains only validated, response-free
diagnostic results.
