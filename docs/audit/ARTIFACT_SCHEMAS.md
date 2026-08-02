# Audit Artifact Schemas

Every audit run produces a manifest matching
`schemas/audit_run_manifest.schema.json`. Large arrays remain on approved
MIND/CORTEX storage; source repositories contain only compact sanitized
manifests, configurations, tests, and reports.

## Required identities

- repository commit and dirty-worktree state;
- estimator contract and numerical mode;
- complete environment and hardware identity;
- logical input IDs, semantic array identities, and cryptographic hashes;
- exact inner and outer split IDs;
- preprocessing, regularization, solver, score, and seed configuration;
- requested and observed resources;
- output artifact hashes; and
- validation, termination, and cleanup status.

## Privacy boundary

Public or shared source artifacts use logical IDs such as
`nsd_derived/subject_01/betas_average`. They must not contain:

- raw neural responses;
- participant-private metadata;
- credentials or tokens;
- absolute MIND/CORTEX paths;
- private host configuration; or
- provider operational records.

The private inventory maps logical IDs to machine-specific paths. It is not
committed to this repository.

## Result comparison

A comparison record names two complete run manifests and reports differences
for each intermediate quantity. A final score comparison alone is not
sufficient for CPU/GPU parity.
