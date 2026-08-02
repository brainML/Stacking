#!/usr/bin/env python3
"""Read-only filesystem inventory for private NSD lineage manifests.

The script writes JSON to stdout and never mutates the inspected tree. Absolute
paths are excluded so its output can be sanitized before sharing. Hashing is
opt-in because large NSD trees require a separately approved hash plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def inventory(root: Path, logical_root: str, hash_mode: str, max_files: int):
    if not root.is_dir():
        raise ValueError("inventory root must be an existing directory")
    records = []
    total_bytes = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories.sort()
        filenames.sort()
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            relative = path.relative_to(root).as_posix()
            record = {
                "logical_id": f"{logical_root}/{relative}",
                "bytes": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }
            if hash_mode == "all":
                record["sha256"] = sha256(path)
            records.append(record)
            total_bytes += metadata.st_size
            if max_files and len(records) >= max_files:
                return records, total_bytes, True
    return records, total_bytes, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--logical-root", required=True)
    parser.add_argument("--hash", choices=("none", "all"), default="none")
    parser.add_argument("--max-files", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.max_files < 0:
        parser.error("--max-files must be nonnegative")
    try:
        records, total_bytes, truncated = inventory(
            arguments.root, arguments.logical_root, arguments.hash, arguments.max_files
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": 1,
                "logical_root": arguments.logical_root,
                "hash_mode": arguments.hash,
                "file_count": len(records),
                "total_file_bytes": total_bytes,
                "truncated": truncated,
                "files": records,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
