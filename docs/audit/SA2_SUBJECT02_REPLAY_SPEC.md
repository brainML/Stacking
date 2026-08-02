# SA2 Participant-2 Legacy Replay Specification

Date: 2026-08-02

Status: **full-hash preflight passed; RNG gate blocks compute**.

This is the first bounded NSD replay candidate. It targets participant 2,
forward structured variance partitioning index 0 (all seven experts), and
cortical voxel chunk 1 (columns 0 through 4,999). It is non-spatial and writes
to a fresh audit namespace.

## Frozen identities

- Historical script SHA-256:
  `dc5b23edccd09dc640f94ee8513180f111e82366e9096e4f82e15c5defc1de36`.
- Response generation: older cortical tree, shape `(10000, 107860)`, float64.
- Feature family: seven pooled AlexNet matrices, float32.
- PCA dimension: 512 per expert.
- Full hashes and byte sizes: recorded in
  [`SA1_CODE_AND_RESULT_PROVENANCE.md`](SA1_CODE_AND_RESULT_PROVENANCE.md).
- Response and feature rows: metadata order, independently checked against
  trial averages and the 73,000-image source matrices.

## Recovered execution behavior

The script:

1. fits a separate `PCA(n_components=512)` to each complete 10,000-image
   feature matrix before encoding-model cross-validation;
2. uses five contiguous outer folds of 2,000 images;
3. standardizes outer-train and outer-test partitions independently;
4. selects per-voxel ridge lambda with four contiguous inner `KFold` splits;
5. searches `10**i` for `i = -6, ..., 9`;
6. fits primal ridge experts and solves one CVXOPT simplex QP per voxel;
7. averages mixture weights and training scores over outer folds; and
8. scores concatenated held-out predictions against the original 10,000-row
   response array using the historical signed-R2 function.

These behaviors belong only to `legacy_reproduction`. In particular, global
PCA, independently fitted test scaling, and contiguous non-grouped folds are
not permitted in `corrected_v2`.

## Existing chunk-1 acceptance artifacts

| Output | Shape | SHA-256 |
| --- | --- | --- |
| `r2s` | `(7, 5000)` | `e3eab7c036df35b81041f3f05970c30a68e42d82c3dfd7504852c14c95d6e426` |
| `stacked_r2s` | `(5000,)` | `b844377d8196b2874ee34da0bec58573d83943d821d4b77dbe95fde29c6c79df` |
| `r2s_weighted` | `(7, 5000)` | `15bb0933b3b8aa55b5b2983e03c701362ab1c2a38d6cc72bc3335c0106454a5d` |
| `r2s_train` | `(7, 5000)` | `6be5f0b3ae7d5f22ac41726c396db5a54ff8e26785cace33e72ce14606506d0e` |
| `stacked_train` | `(5000,)` | `f39934931eaf819e21a39f5b6a73e291cae0a06693c03b5f7710e6ab6ecec68a` |
| `S_average` | `(5000, 7)` | `aab8da8170dedb209d51eafac15967b04706a0344862f293f4d458ea8bcc4c66` |

## Stochastic-PCA blocker

The recovered script sets neither `random_state` on PCA nor a global NumPy
seed. With 10,000 rows, thousands of columns, and 512 components,
scikit-learn 0.23.2 selects its randomized PCA path under `svd_solver="auto"`.
The exact historical PCA draws were not saved.

Consequently, a fresh run may fail byte-level and strict numerical comparison
even if every later operation is faithfully reconstructed. The audit must not
call such a run an exact reproduction.

The resolution order is:

1. search for a saved PCA realization, job wrapper seed, environment capture,
   or RNG state;
2. if recovered, run the exact environment and compare all six hashes;
3. otherwise run a small prospectively frozen PCA-seed sensitivity panel and
   quantify how much the six outputs vary;
4. select and record one seed only for a deterministic
   `legacy_reconstruction` variant; and
5. keep the original artifacts as the sole `published_historical` tier.

The current MIND login environment reports Python 3.8.5, scikit-learn 0.23.2,
NumPy 1.23.0, and SciPy 1.5.2. This observation is not proof that the same
environment generated the historical artifacts.

## Launch gate

Before running the 11-GB packet, the audit still requires:

- a manifest-driven runner that cannot overwrite the historical directory;
- a dry-run mode validating every full input hash, shape, and dtype;
- an explicit PCA seed/RNG classification;
- captured Python/package/BLAS/CVXOPT identities;
- predicted RAM, scratch, and wall-time bounds; and
- an output validator that compares every intermediate available, not only
  the final stacked score.

GPU and Lambda execution remain out of scope for this legacy CPU replay.

The sanitized full-hash preflight record is
[`SA2_SUBJECT02_PREFLIGHT.json`](SA2_SUBJECT02_PREFLIGHT.json). It passed every
data and overwrite-safety check while correctly reporting
`rng_gate.launch_allowed: false`. No directory or artifact was created on
MIND.
