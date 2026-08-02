# Conditional GPU Full-Feature Arm

Date: 2026-08-02

Status: **planned; blocked on corrected-v2 CPU and GPU float64 parity**.

The audit will preserve the historical PCA pipelines exactly under
`legacy_reproduction`. If the GPU path clears its numerical and operational
gates, `corrected_v2` will add a separate full-feature arm that fits the ridge
experts without PCA.

## Scientific comparison

The corrected analysis will freeze two feature-transform policies before
reading outcomes:

1. `nested_pca`: PCA is fitted only on each training partition and applied
   unchanged to validation/test rows.
2. `none_full_feature`: the original feature coordinates are retained; only
   train-fitted centering/scaling is permitted.

Both arms reuse identical response rows, acquisition masks, outer/inner
splits, lambda candidates, score definitions, and stacking queries. The
comparison estimates the effect of PCA rather than confounding it with folds,
preprocessing, GPU precision, or feature-generation lineage. No-PCA is a
candidate scientific condition, not an assumed improvement.

## Required numerical design

The current GPU prototype forms `X.T @ X`, which is unsuitable for full
AlexNet maps with hundreds of thousands of columns. The full-feature arm must:

- choose primal or dual ridge from the actual `(n_samples, n_features)` shape;
- use the dual/kernel form when `n_features >> n_samples` and never materialize
  a `p x p` covariance matrix;
- reuse a reviewed eigendecomposition/factorization across lambda candidates;
- stream feature blocks from host storage to construct centered/scaled Gram
  matrices rather than requiring every layer to reside on device at once;
- process feature experts sequentially and persist only the required
  out-of-fold/test predictions;
- chunk target voxels with invariant outputs and at least 20% measured VRAM
  headroom; and
- save solver residuals, selected lambdas, dtype, peak RAM/VRAM, and timing.

For a training matrix `X` with `p >> n`, predictions should be obtained through
the dual system

```text
(X X^T + lambda I) alpha = Y
prediction = X_test X^T alpha
```

or a numerically equivalent reviewed formulation. The implementation must not
materialize full primal weights unless a bounded diagnostic explicitly needs
them.

## Gate sequence

1. Prove corrected CPU primal/dual equivalence on deterministic fixtures.
2. Prove PyTorch CPU float64 parity at every saved intermediate.
3. Prove CUDA float64 parity and chunk invariance on synthetic `p >> n` cases.
4. Run a data-local NSD micro-pilot with two layers and a small voxel set.
5. Benchmark `nested_pca` and `none_full_feature` under identical splits.
6. Consider Lambda only after scheduler validation, data authorization, and a
   favorable end-to-end time/cost decision.

The no-PCA arm cannot be used to reproduce a published PCA result, and its
outputs must use a distinct transform-policy ID and artifact namespace.
