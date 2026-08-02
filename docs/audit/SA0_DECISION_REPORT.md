# SA0 Decision Report

Date: 2026-08-02

Decision: **SA0 passes for the bounded audit packet. SA1 is active.**

## Frozen foundation

- Historical CPU baseline: `2232c0db521fd37aecb24eea63c126ff3f4d0adf`.
- GPU prototype baseline: `d9eeeca4f685b374e04ce2a324cc7d6323a9133c`.
- Scheduler baseline: `b4234c62ea2bd12841a48bd22cb3a3334bafc78f`.
- Program roadmap baseline: `cc56922997e5f21e05f7688d4d864ae5061c974a`.
- Existing CPU regression suite: five tests passing in the recorded local
  Anaconda environment.

The pre-existing untracked CPU `__pycache__/` directory was recorded and left
untouched.

## Current authorization

Permitted:

- local contract, schema, fixture, and behavioral-test work;
- read-only MIND/CORTEX discovery and metadata inventory; and
- private generation of machine-specific lineage manifests.

Blocked:

- scientific NSD replay;
- full-file hashing of unadjudicated large trees;
- transfer of NSD neural arrays to Lambda;
- Lambda instance creation; and
- use of `Stacking_GPU` for a scientific result.

## Exit criterion for SA1

SA1 passes only when the candidate MIND and CORTEX derivatives have explicit
logical identities, participant/image/voxel/feature semantics, and sufficient
hash evidence to classify them as mirrors, compatible derivatives, conflicts,
incomplete, or quarantined. Location and modification time alone do not confer
authority.
