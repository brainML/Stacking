# SA1 Read-Only NSD Inventory

Date: 2026-08-02

Status: **partial pass; a mixed code/result lineage conflict is confirmed**.

No remote file was created, modified, or transferred. The inventory read
directory metadata and NumPy headers. One small stimulus metadata file was
hashed. Large neural and feature arrays were not recursively hashed.

## MIND findings

MIND contains the legacy derivatives previously associated with the published
and follow-up code.

### Eight-participant averaged cortical responses

All arrays are float64 with 10,000 stimulus rows:

| Participant | Shape |
| --- | --- |
| 01 | `(10000, 107104)` |
| 02 | `(10000, 107860)` |
| 03 | `(10000, 111068)` |
| 04 | `(10000, 101628)` |
| 05 | `(10000, 94578)` |
| 06 | `(10000, 120668)` |
| 07 | `(10000, 90214)` |
| 08 | `(10000, 108767)` |

The same tree contains z-scored trial/session arrays for all eight
participants. Participants 01, 02, 05, and 07 have 30,000 rows; 03 and 06 have
24,000; 04 and 08 have 22,500. Unstandardized trial/session arrays were found
for participants 01 and 05 only.

### Legacy feature panel

Feature directories exist for all eight participants plus a separate
participant-1 revision directory. The pooled AlexNet arrays for participants
1, 2, 5, and 7 are float32 with 10,000 rows:

| Representation | Columns |
| --- | ---: |
| conv1 | 9,216 |
| conv2 | 9,408 |
| conv3 | 9,600 |
| conv4 | 9,216 |
| conv5 | 9,216 |
| fc6 | 4,096 |
| fc7 | 4,096 |

Participants 3, 4, 6, and 8 instead contain full spatial convolutional maps:
`(64, 55, 55)`, `(192, 27, 27)`, `(384, 13, 13)`, `(256, 13, 13)`, and
`(256, 13, 13)`, followed by 4,096-dimensional `fc6` and `fc7`. Thus the
recovered cohort differs in pre-PCA feature construction as well as PCA
dimension.

The directory also includes ResNet, Taskonomy, caption-trained, place-trained,
and other legacy representations. Checkpoint identity remains unresolved. The
pooled participant-1 and participant-2 feature rows match deterministic
samples from the located 73,000-image source matrices exactly; other
participants remain to be checked or require a different raw-map source.

### Stimulus metadata

The candidate merged stimulus metadata file is 11,310,098 bytes with SHA-256:

```text
e46fca15901196541b8921aa8c03ee9d662e7da085d852e13ebb01f89c7623c6
```

Its participant membership and repeat-position semantics now reproduce the
row order and sampled numerical values of all eight averaged response arrays.
The relationship to an independently obtained official NSD metadata release
still requires validation.

For participant 1, the three repeat-position columns identify 10,000 unique
images and form an exact one-based partition of trial positions 1 through
30,000. Converting these positions to a trial-ordered int64 NSD-ID sequence
produces SHA-256:

```text
d86a79a8f06ebb57616eda68c8393c966996c8aed80f87d18d0ff13b3d217671
```

For participants 1, 2, 5, and 7, sampled averaged responses exactly equal the
mean of the three trial responses. Participants 3 and 6 contain only 24,000
trials, and participants 4 and 8 contain 22,500. Their averaged arrays use the
sum of acquired responses divided by a fixed three: respectively 589 and 791
images have no acquired repeat and sampled values for those rows are exactly
zero. This convention is part of the legacy lineage, not the corrected data
contract.

## CORTEX findings

CORTEX contains two distinct subject-1-oriented NSD trees rather than a mirror
of the MIND eight-participant derivatives.

### `NSD2` derivative tree

The tree occupies approximately 8.2 GiB and includes:

- 73,000-image float32 feature matrices for GIST, ResNet-50, VGG16, expanded
  VGG16, and Xception;
- participant-1 session-arranged feature arrays with shape
  `(40, 750, feature_dim)`;
- a 30,000-element int64 stimulus-index array;
- session-level participant-1 response derivatives; and
- extraction and subject-reading scripts whose provenance has not yet been
  audited.

The participant-1 stimulus-index array contains 10,000 unique NSD IDs, and its
canonical int64 sequence hash exactly matches the sequence derived
independently from the MIND metadata. This closes the trial-order identity for
that one artifact. It does not yet establish the row order of the MIND averaged
response or feature arrays.

### Subject-1 beta tree

An approximately 125 GiB subject-1 tree contains sessionwise 1.8 mm beta,
response-quality, HRF, and related derivatives. Both HDF5 and NIfTI variants
are present. NSD-synthetic outcome files also exist in this tree; their
contents were not inspected and remain sealed for downstream confirmation.

## Cross-system classification

| Candidate | Classification | Reason |
| --- | --- | --- |
| MIND eight-participant cortical averages | Unverified legacy derivative | Complete participant span, but official stimulus/voxel lineage unresolved |
| MIND legacy feature panel | Unverified historical feature derivative | Architecture labels exist; checkpoint, extraction, order, and PCA lineage unresolved |
| CORTEX `NSD2` | Lineage-compatible candidate, not a mirror | Subject-1 session-oriented structure and different feature families/dimensions |
| CORTEX subject-1 beta tree | Official-style source candidate | Rich session data, but only participant 1 was located and release identity is unverified |
| MIND versus CORTEX | Not hash-comparable as trees | Different participant coverage, shapes, representations, and derivative levels |

## Historical code candidates on CORTEX

Two pre-existing repositories were found and left untouched:

- a `Stacking` worktree at commit
  `f13e896ae322f93c9796de3aacdf80ff6a7a787c` from 2024-06-05. This commit is
  an ancestor of the current public repository. Its worktree has a modified
  ridge utility and untracked simulation scripts, arrays, and figures;
- a `Stacking_OLD` worktree at commit
  `7edffc7b946fe5f60ff23830dafe845f1c4a3b87` from 2020-04-07. It contains
  early NSD notebooks, cortical extraction/ROI scripts, GPU ridge code,
  simulation outputs, and several modified or untracked files.

Neither dirty worktree is an authoritative source by itself. Their tracked
commits, working-tree diffs, notebooks, and untracked analysis artifacts must
be inventoried independently before deciding which code generated each
published result.

The local historical archive resolves part of the implementation question and
confirms a cohort-level conflict. The paper reports 1,024 PCA dimensions per
AlexNet layer. The recovered participant-1 structured-variance scripts use
1,024, but the recovered participant-2/5/7 structured-variance scripts and
participant-3/4/6/8 ordinary stacking scripts use 512. All fit PCA to all
10,000 images before encoding-model cross-validation.

Result chunk lengths close exactly against the identified response arrays for
participants 1, 2, 3, 4, 6, and 8. Participants 5 and 7 instead close against
the older cortical tree and its recovered masks, differing from the newer tree
by 83 and 30 voxels respectively. See
[`SA1_CODE_AND_RESULT_PROVENANCE.md`](SA1_CODE_AND_RESULT_PROVENANCE.md) for
the script, mask, dirty-worktree, and representative result hashes. The
published cohort cannot presently be described as one 1,024-component,
single-response-generation pipeline.

## Storage observation

The MIND NSD data volume is 97% occupied with approximately 38 GiB free, while
the MIND home volume is 98% occupied. CORTEX has approximately 1.7 TiB free on
its local root filesystem, but its home NAS is 95% occupied. No audit plan may
assume that the large CORTEX local disk is an approved persistent project store
until that policy is confirmed.

## Remaining SA1 work

1. Link every published panel to an exact result-family hash and executable
   script; preserve the now-confirmed participant-specific PCA/response
   lineages rather than collapsing them.
2. Close feature-row identity by comparing subject feature rows with the
   identified source matrix; response-row identity is now numerically verified
   for participants 1-8.
3. Map cortical columns to official voxel spaces and ROI masks.
4. Establish checkpoint, PCA dimension, PCA fit partition, and preprocessing
   provenance for the seven AlexNet feature spaces.
5. Select a small list of critical files for full SHA-256 comparison; do not
   recursively hash the 100+ GiB trees.
6. Mark NSD-synthetic paths as sealed in all private manifests and inventory
   tooling.
7. Preserve and separately hash the tracked commits, dirty diffs, and relevant
   untracked artifacts in the two historical CORTEX code worktrees.

Until these items close, MIND is the historical replay source and CORTEX is a
subject-1 lineage source and possible compute location. Neither is a verified
authoritative mirror of the other.
