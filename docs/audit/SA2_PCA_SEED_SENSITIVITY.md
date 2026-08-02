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

The diagnostic is configured in
`configs/audit/pca_seed_sensitivity_subject02_conv3.json`. It requests one
Slurm CPU job, 16 CPUs, 64 GiB RAM, four hours, and no concurrency. The bounded
feature-only job completed on 2026-08-02 in the recovered environment.
Scheduler identifiers, cluster paths, and logs remain in the ignored private
run manifest; the public record contains only validated, response-free
diagnostic results.
