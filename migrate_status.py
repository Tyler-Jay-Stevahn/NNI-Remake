#!/usr/bin/env python3
"""migrate_status.py — collapse proposal status to a single 4-value field.

Maps the legacy two-field lifecycle (status: proposed/approved/tested +
compile_status: untested/ok) into one `status` field:
    tested                       -> trained
    compile_status == "ok"      -> compiles
    everything else             -> proposed

Drops the now-redundant `compile_status` and `approved` keys from every
record. Idempotent: running it again is a no-op on already-migrated data.

Usage:
    python3 migrate_status.py            # rewrite proposals.jsonl in place
    python3 migrate_status.py --dry      # print changes, don't write
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "proposals.jsonl")

VALID = {"proposed", "compiles", "fails", "trained"}


def migrate(rec):
    old_status = rec.get("status")
    old_compile = rec.get("compile_status")

    # Already on the modern 4-value ladder: keep it (idempotent on re-run).
    if old_status in VALID:
        new_status = old_status
    elif old_status == "tested":
        new_status = "trained"
    elif old_compile == "ok":
        new_status = "compiles"
    else:
        new_status = "proposed"

    rec["status"] = new_status
    rec.pop("compile_status", None)
    rec.pop("approved", None)
    return new_status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print changes, don't write")
    args = ap.parse_args()

    with open(SRC, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    counts = {}
    for rec in records:
        s = migrate(rec)
        counts[s] = counts.get(s, 0) + 1

    if args.dry:
        print("DRY RUN — would write:")
        for k in sorted(counts):
            print(f"  {k}: {counts[k]}")
        return

    with open(SRC, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print("Migrated. New status counts:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
