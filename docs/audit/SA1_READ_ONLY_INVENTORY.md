# SA1 Read-Only NSD Inventory

Date: 2026-08-02

Status: **partial pass; lineage adjudication remains open**.

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
participant-1 revision directory. The participant-1 AlexNet arrays are
float32 with 10,000 rows:

| Representation | Columns |
| --- | ---: |
| conv1 | 18,496 |
| conv2 | 19,200 |
| conv3 | 18,816 |
| conv4 | 16,384 |
| conv5 | 16,384 |
| fc6 | 4,096 |
| fc7 | 4,096 |

The directory also includes ResNet, Taskonomy, caption-trained, place-trained,
and other legacy representations. Checkpoint, extraction, stimulus-order, and
PCA provenance are not yet established by filenames or headers.

### Stimulus metadata

The candidate merged stimulus metadata file is 11,310,098 bytes with SHA-256:

```text
e46fca15901196541b8921aa8c03ee9d662e7da085d852e13ebb01f89c7623c6
```

Its row semantics and relationship to the 10,000-row participant arrays still
require validation against official NSD identities.

For participant 1, the three repeat-position columns identify 10,000 unique
images and form an exact one-based partition of trial positions 1 through
30,000. Converting these positions to a trial-ordered int64 NSD-ID sequence
produces SHA-256:

```text
d86a79a8f06ebb57616eda68c8393c966996c8aed80f87d18d0ff13b3d217671
```

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

The local historical archive adds another unresolved implementation question.
The paper reports 1,024 PCA dimensions per AlexNet layer, while readily located
eight-participant scripts use 512 dimensions. A few older subject-1 scripts use
1,024 dimensions, but target different source paths or model variants. All of
these scripts fit PCA to all 10,000 images before outer cross-validation. The
exact scripts and PCA realization used for the published cohort therefore
remain unidentified; the located code cannot yet be treated as the published
pipeline.

## Storage observation

The MIND NSD data volume is 97% occupied with approximately 38 GiB free, while
the MIND home volume is 98% occupied. CORTEX has approximately 1.7 TiB free on
its local root filesystem, but its home NAS is 95% occupied. No audit plan may
assume that the large CORTEX local disk is an approved persistent project store
until that policy is confirmed.

## Remaining SA1 work

1. Identify the exact published NSD scripts, masks, PCA objects, and stimulus
   selection/order artifacts.
2. Extend the now-verified participant-1 trial sequence mapping to the row
   order of the MIND averaged responses/features and to participants 2-8.
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
