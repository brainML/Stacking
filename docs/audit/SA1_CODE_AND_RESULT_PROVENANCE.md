# SA1 Code and Result Provenance

Date: 2026-08-02

Status: **conflict found; scientific replay remains blocked**.

This report links archived executable scripts to the legacy MIND result
directories using literal input/output paths, PCA settings, output naming,
NumPy headers, and a bounded set of SHA-256 hashes. It does not claim that an
artifact was used in a published figure unless a figure notebook points to it.

No remote file was created, modified, or transferred. Large response, feature,
and result trees were not recursively hashed.

## Cohort-level conflict

The paper describes reducing every AlexNet feature space to 1,024 principal
components. The recovered scripts and result directories instead identify two
PCA regimes, two pre-PCA convolutional representations, and at least two
response generations:

| Participant | Candidate result family | PCA in matching script | Response generation in matching script | Concatenated result voxels | Available response voxels | Classification |
| --- | --- | ---: | --- | ---: | --- | --- |
| 01 | `subj1_stack_alexnet_vp_1024` | 1,024 | separate participant-1 legacy path | 107,104 | 107,104 in current legacy trees | dimension-compatible; exact response file unresolved |
| 02 | `subj2_stack_alexnet_vp_512` | 512 | older cortical tree | 107,860 | old 107,860; new 107,860 | dimension-compatible; generation not distinguishable by length |
| 03 | `subj3_stack_alexnet_512` | 512 | newer cortical tree | 111,068 | new 111,068 | exact dimension match |
| 04 | `subj4_stack_alexnet_512` | 512 | newer cortical tree | 101,628 | new 101,628 | exact dimension match |
| 05 | `subj5_stack_alexnet_vp_512` | 512 | older cortical tree | 94,495 | old 94,495; new 94,578 | exact old-tree match; conflicts with new tree by 83 voxels |
| 06 | `subj6_stack_alexnet_512` | 512 | newer cortical tree | 120,668 | new 120,668 | exact dimension match |
| 07 | `subj7_stack_alexnet_vp_512` | 512 | older cortical tree | 90,184 | old 90,184; new 90,214 | exact old-tree match; conflicts with new tree by 30 voxels |
| 08 | `subj8_stack_alexnet_512` | 512 | newer cortical tree | 108,767 | new 108,767 | exact dimension match |

For participants 5 and 7, the result lengths also equal the nonzero counts in
the recovered older cortical masks. This is positive evidence that those
results were computed in the older voxel space. It is not a filename-only
inference.

The cohort therefore cannot currently support an “all participants, 1,024
components, one response preprocessing generation” reproduction claim. The
legacy artifacts remain valid inputs to the `legacy_reproduction` contract,
but they must be represented as participant-specific lineages. A corrected
cohort must use one declared feature/PCA/response policy and new result IDs.

Participants 1, 2, 5, and 7 use spatially pooled convolutional matrices with
9,216/9,408/9,600/9,216/9,216 columns. Participants 3, 4, 6, and 8 use full
AlexNet maps with shapes `(64, 55, 55)`, `(192, 27, 27)`, `(384, 13, 13)`,
`(256, 13, 13)`, and `(256, 13, 13)` before flattening. Giving both groups 512
PCA components does not erase this difference in their PCA inputs.

## Response row identity and incomplete acquisitions

The audit reconstructed response averages from the trial arrays using the
repeat positions in the hashed NSD metadata. It sampled metadata-spanning image
rows and cortical columns without copying the source arrays.

| Participant | Trial rows | Response generation tested | Images with no acquired repeat | Rule matching every sampled value |
| --- | ---: | --- | ---: | --- |
| 01 | 30,000 | newer | 0 | mean of 3 acquired repeats |
| 02 | 30,000 | older | 0 | mean of 3 acquired repeats |
| 03 | 24,000 | newer | 589 | sum of acquired repeats divided by 3 |
| 04 | 22,500 | newer | 791 | sum of acquired repeats divided by 3 |
| 05 | 30,000 | older | 0 | mean of 3 acquired repeats |
| 06 | 24,000 | newer | 589 | sum of acquired repeats divided by 3 |
| 07 | 30,000 | older | 0 | mean of 3 acquired repeats |
| 08 | 22,500 | newer | 791 | sum of acquired repeats divided by 3 |

For participants 3, 4, 6, and 8, sampled rows with one or two acquired repeats
match a fixed three-repeat denominator with zero numerical error. Sampled rows
with no acquired repeat are exactly zero. Because the recovered scripts load
all 10,000 rows, these 589 or 791 zero-response images were available to model
fitting rather than being excluded as unobserved.

This closes the averaged-response **row order to the audit's deterministic
sampling standard**: it is the metadata row order of the participant's 10,000
NSD images. It also identifies a material legacy preprocessing behavior. A
corrected analysis must represent acquisition
availability explicitly, average only acquired repeats, and exclude images
with no response from fitting and scoring. It must not relabel the legacy
fixed-denominator arrays as corrected inputs.

The recovered feature-subsetting script selects the same metadata rows in the
same order. For participants 1 and 2, all seven pooled feature files match the
located 73,000-image source matrices exactly across deterministic samples of
32 metadata-spanning rows and 16 columns per layer. Their source-row index
hashes equal their metadata image-order hashes. This closes participant 1 and
2 feature-row identity to the audit sampling standard. Participants 5 and 7
can be checked against the same pooled source; participants 3, 4, 6, and 8
require the still-unidentified full-map source. Both recovered copies of the
subsetting script have SHA-256
`cfac59bb20752956363fb420474e5b5418fad6b2519f2aa9ad18b3bcf22a4bd1`.

The historical participant-1 response path named by the 1,024-component
scripts no longer exists. The surviving older and newer participant-1 arrays
have the same shape but differ on deterministic samples (maximum sampled
absolute difference 0.05926452737376545). Participant 1 therefore remains an
ambiguous numerical replay target despite its closed feature rows. Participant
2 is the preferred first replay target because its script-named older response
array, pooled feature family, and result directory all survive.

### Frozen participant-2 replay inputs

The preferred one-chunk replay packet has full-file SHA-256 identities:

| Logical input | Bytes | SHA-256 |
| --- | ---: | --- |
| older averaged cortical responses | 8,628,800,128 | `3414a33e03fc883e9d290dc57cb5da6d713a9141ab28c54258524ffb820a1f1f` |
| pooled AlexNet conv1 | 368,640,128 | `9d2f981893075a7df6fb2a10ecd70f8a7c1958dcb5411475a1d1dd23144f3b9c` |
| pooled AlexNet conv2 | 376,320,128 | `01454fe4feb31597682282f9a727471d12488c0252d60ad6b9b667534b6d03eb` |
| pooled AlexNet conv3 | 384,000,128 | `61449e0ddf74e1e18677dc5f00706ccb9b05318c426d3a497fb21c6345872d46` |
| pooled AlexNet conv4 | 368,640,128 | `64c02ed12b80c36ac2a11a3bc916d74e54e769a580015c712cd72d0d2a6d8a58` |
| pooled AlexNet conv5 | 368,640,128 | `eef96f681c44d231f6fc2d17770229b21e8c1adaea02127e7ec3420d0fde1461` |
| pooled AlexNet fc6 | 163,840,128 | `5e308b38ebe0eda47cd71a77395a48303c11ae114357a6a7be920638e1bf4fc0` |
| pooled AlexNet fc7 | 163,840,128 | `66169373717e11802d4a122d848331245910af2854823b3092d57788c5da444d` |

These are the only large arrays full-hashed in this SA1 step. The remaining
trees were not recursively hashed.

## Executable script evidence

The recovered forward/reverse structured-variance scripts have these SHA-256
identities:

| Logical role | SHA-256 |
| --- | --- |
| participant 1, forward, 1,024 components | `fcefb7845f2456df3a631d2e7b745338c10d7f7f78249586f77e88e8cbd3d4a5` |
| participant 1, reverse, 1,024 components | `c6fcf62ff74a6daf99869ddaf11e24a8738a2148904c11172e559fd2d00280ce` |
| participant 2, forward, 512 components | `dc5b23edccd09dc640f94ee8513180f111e82366e9096e4f82e15c5defc1de36` |
| participant 2, reverse, 512 components | `ea89eb43225156e34785c99e2d993b6c3a70eac32a573225a26c9b20a08aa0b5` |
| participant 5, forward, 512 components | `180f411bed05de87f1f881d82a35a4a72e1847ccdcdeaffaade63ec2842feec6` |
| participant 5, reverse, 512 components | `54815f5659bb6160c98caa481655bf268694dcd06f92c7792fe650c6407c8e89` |
| participant 7, forward, 512 components | `202d300f32f895d352644584da710c871bae15f7df8a035129bddc0a5257f630` |
| participant 7, reverse, 512 components | `60bfed753bfefa4838f8de8b6730c22aa867054c4138866b804681de04c50cc1` |

The recovered ordinary seven-expert scripts for participants 3, 4, 6, and 8
use 512 components and the newer cortical tree. Their hashes are respectively:

```text
d05ae320151d57c38a56c06c14fbe86e2b7ad86a552248fe2c7f69360c405c5b
ff166cdf977a21e7f08de6ace8ea99758192f375031ea45e1cfe9ead6c0f1e99
518c7969de601efdb265ecb352c90bb8fae330ce6e4dba378de3ace9768e0b2b
d98fa1c3213f437ea5bf1789942fdedac2e18da4a0982390ac713750e0207c6d
```

All recovered scripts call `PCA.fit_transform` on each complete 10,000-image
feature matrix before the encoding-model cross-validation. That behavior is
part of `legacy_reproduction`; it is disallowed by `corrected_v2`, where the
transform must be fitted inside the applicable training partition.

## Result structure and bounded hashes

Each inspected stacking-weight chunk is float64 with shape `(n_voxels, 7)`.
Each inspected stacked-score chunk is a float64 vector. Chunks are nominally
5,000 voxels, with one shorter terminal chunk per participant. The total
lengths in the table above close exactly against the identified response
generation.

Representative first-chunk stacked-score hashes:

| Participant | SHA-256 |
| --- | --- |
| 01 | `ca2f8227921b39053ae3eea976fd61572c0761fa09c5dd604ca93e5c08589c52` |
| 02 | `b844377d8196b2874ee34da0bec58573d83943d821d4b77dbe95fde29c6c79df` |
| 03 | `48e2f035215c69b87255af414fc2307e03f9c5cb19bd8b717ce740df7f7a9ac4` |
| 04 | `d9b5c44c9a6cb18e774c84b4896b55372399c17cfe17370698fc2d829bfc45ef` |
| 05 | `9704fa2a6971820526dd03bf9cd8d8d6292a30b7b089c74a4ec88b6ea722f0f4` |
| 06 | `19a9ae4f6624399c5d42386b26312217441847e63e4fd642344be53aa558dc63` |
| 07 | `3ba59fb1e94697e874177cc25fc6ce22bce564c70ba1d0c87854ffb8f16ed721` |
| 08 | `d71c1f1ceb542d15b554df8fc28e3a53b5b808ec70f696d221fa12b445a70ff9` |

The participant-1 variance-partitioning figure notebook explicitly reads the
1,024-component result family. No recovered figure source yet establishes
which result family supplied every panel and participant in the published
paper.

## Mask evidence

The two recovered old-space masks are immutable candidates for voxel-to-volume
mapping:

| Participant | Shape | Nonzero voxels | SHA-256 |
| --- | --- | ---: | --- |
| 05 | `(79, 97, 78)` | 94,495 | `807e8019f15b3cab05189bb4044a441dfc5d3e286d7b3113b9bf111c1d0c8198` |
| 07 | `(78, 95, 81)` | 90,184 | `b40285e487cd00a25601395ec5cba561d931d08b22a352ba4910c18f245eb7ad` |

Equivalent masks or index maps for the other participants have not been
located. Cortical result vectors must not be projected into a volume or
surface using a newly inferred ordering.

## Historical worktree identities

The two dirty CORTEX worktrees remain read-only. Their full binary-diff hashes
are:

```text
Stacking:     93bae0dc6a565a184f14be663cb152fec4b67c55280a9f4dbdaf9b476a5d8522
Stacking_OLD: 9e6419780751607d0be7058d73e71ecb6adb2e57697915dee299d39ecafddfbe
```

These hashes identify the observed dirty states; they do not make either state
authoritative. The tracked commits remain separately identified in the SA1
inventory.

## Replay gate

A bounded CPU legacy replay may begin only after its manifest declares:

1. one participant and one exact response generation;
2. the exact feature family and PCA dimension;
3. the recovered script hash or an explicitly documented behavioral port;
4. the row-order identity used for both features and responses;
5. the cortical mask/index identity, if spatial output is requested; and
6. a fresh output namespace that cannot overwrite legacy results.

The first replay should be a non-spatial, one-chunk participant-1 or
participant-3 test. GPU parity and Lambda execution remain downstream gates,
not substitutes for lineage resolution.
