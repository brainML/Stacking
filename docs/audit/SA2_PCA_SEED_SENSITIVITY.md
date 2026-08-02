# SA2 Response-Free PCA Seed Sensitivity

Date: 2026-08-02

Status: **frozen plan; not submitted**.

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

## Escalation rule

If the three full-matrix PCA runs are effectively invariant at prespecified
numerical tolerances, freeze one seed for a labeled deterministic legacy
reconstruction. If they are materially different, run all seven experts for
the same three seeds but restrict downstream evaluation to one outer fold and
256 prospectively fixed voxels. Do not search a large seed space for the seed
that best matches the published outcomes.

The diagnostic is configured in
`configs/audit/pca_seed_sensitivity_subject02_conv3.json`. It requests one
Slurm CPU job, 16 CPUs, 64 GiB RAM, four hours, and no concurrency. Submission
is deferred because the account already has substantial unrelated queued work;
the audit will not contend with it without an explicit scheduling decision.
