#!/usr/bin/env python3
"""Renumber cam folders in a dataset to sequential camXX names.

Example:
  python scripts/renumber_cams.py --dataset data/multipleview/coffee_martini --dry-run
  python scripts/renumber_cams.py --dataset data/multipleview/coffee_martini

The script finds directories matching the pattern <prefix><number> (default prefix "cam"),
sorts them by their numeric suffix, and renames them to a consecutive sequence
starting from `--start-index` using zero-padded numbering (width chosen automatically).

It performs safe renames using temporary names to avoid collisions.
"""

import re
import os
import argparse
from pathlib import Path


def find_cam_dirs(root: Path, prefix: str):
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)$')
    cams = []
    for p in root.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                cams.append((int(m.group(1)), p))
    cams.sort(key=lambda x: x[0])
    return cams


def safe_renames(pairs, dry_run=False):
    # pairs: list of (old_path, new_path)
    # Step 1: rename old -> tmp_%03d_old to avoid conflicts
    tmp_names = []
    for i, (old, new) in enumerate(pairs):
        parent = old.parent
        tmp = parent / (f".__renametmp_{i}__" + old.name)
        tmp_names.append((old, tmp, new))
        if dry_run:
            print(f"DRY: {old} -> {tmp}")
        else:
            print(f"Renaming {old} -> {tmp}")
            old.rename(tmp)

    # Step 2: rename tmp -> final
    for old, tmp, final in tmp_names:
        if dry_run:
            print(f"DRY: {tmp} -> {final}")
        else:
            print(f"Renaming {tmp} -> {final}")
            tmp.rename(final)


def main():
    parser = argparse.ArgumentParser(description="Renumber cam folders to sequential names.")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder (e.g. data/multipleview/coffee_martini)")
    parser.add_argument("--prefix", default="cam", help="Folder prefix (default: cam)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for renumbering (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without performing them")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {root}")

    cams = find_cam_dirs(root, args.prefix)
    if not cams:
        print(f"No folders with prefix '{args.prefix}' and numeric suffix found in {root}")
        return

    num_cams = len(cams)
    pad = max(2, len(str(args.start_index + num_cams - 1)))

    print(f"Found {num_cams} camera folders. Will rename to {args.prefix}00..{args.prefix}{num_cams-1:0{pad}d}")

    pairs = []
    for i, (old_index, old_path) in enumerate(cams):
        new_index = args.start_index + i
        new_name = f"{args.prefix}{new_index:0{pad}d}"
        new_path = old_path.parent / new_name
        if old_path.samefile(new_path) if new_path.exists() else False:
            # already the same (rare); skip
            print(f"Skipping (already): {old_path}")
            continue
        pairs.append((old_path, new_path))

    if not pairs:
        print("No renames necessary.")
        return

    print("Planned renames:")
    for old, new in pairs:
        print(f"  {old.name} -> {new.name}")

    if args.dry_run:
        print("Dry run mode - no changes made.")
        safe_renames(pairs, dry_run=True)
        return

    confirm = input("Proceed with renaming? Type 'yes' to continue: ")
    if confirm.strip().lower() != "yes":
        print("Aborted by user.")
        return

    safe_renames(pairs, dry_run=False)
    print("Renaming complete.")


if __name__ == "__main__":
    main()
