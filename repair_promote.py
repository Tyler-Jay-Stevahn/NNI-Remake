#!/usr/bin/env python3
"""
repair_promote.py  (append-only repair -> in-place fix + resubmit)

Reads proposals.jsonl + fails.jsonl. For every ORIGINAL proposal whose
status == 'fails' (and id does NOT end with '-fix1'):
  - preserve the broken definition in `pre_repair` (full row) and
    `failed_definition` (just the broken block code),
  - copy the VERIFIED fixed `spec.blocks` from its `<id>-fix1` sibling,
  - set status -> 'proposed' (resubmit through the compile gate),
  - record `repair_note` + `original_error` for audit.

The now-redundant `<id>-fix1` parking rows are dropped (the fix now lives
in the original, so keeping both would be pure duplication).

Repair/fix proposals are EXEMPT from the unique-block_type rule, so any
block_type collisions among fixed rows are allowed.

No torch is used; every fixed `definition` was already syntax-checked with
compile() when the -fix1 rows were authored. We re-run that check here.
"""
import json, copy, datetime, re

PATH = "proposals.jsonl"
FAILS = "fails.jsonl"
OUT = "proposals.jsonl"  # overwrite in place (we made a backup first)

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

rows = load(PATH)
by_id = {r["id"]: r for r in rows}

# original_error lookup from the gate's failure capture
fail_err = {}
try:
    for l in open(FAILS):
        l = l.strip()
        if not l:
            continue
        d = json.loads(l)
        fail_err[d["id"]] = d.get("error", "")
except FileNotFoundError:
    pass

def class_of(defn):
    m = re.search(r"class\s+(\w+)\s*\(", defn or "")
    return m.group(1) if m else None

targets = [r for r in rows if r.get("status") == "fails" and not r["id"].endswith("-fix1")]
print(f"original 'fails' rows to fix: {len(targets)}")

promoted = 0
dropped = 0
syntax_failed = []

for r in targets:
    rid = r["id"]
    fid = rid + "-fix1"
    if fid not in by_id:
        print(f"  WARN: no -fix1 sibling for {rid}; skipping")
        continue
    fix1 = by_id[fid]

    # preserve the broken architecture for audit ("failed model is left")
    r["pre_repair"] = copy.deepcopy(r)
    # capture the broken block code(s) specifically
    broken = []
    for b in r.get("spec", {}).get("blocks", []):
        if b.get("type") != "embedding" and b.get("definition"):
            broken.append(b["definition"])
    r["failed_definition"] = "\n\n".join(broken)

    # bring over the verified fixed blocks (strip the '-fix1' block_type suffix)
    fixed_blocks = copy.deepcopy(fix1["spec"]["blocks"])
    for b in fixed_blocks:
        bt = b.get("block_type")
        if isinstance(bt, str) and bt.endswith("-fix1"):
            b["block_type"] = bt[: -len("-fix1")]
    r["spec"]["blocks"] = fixed_blocks

    # resubmit
    r["status"] = "proposed"
    r["repaired"] = True
    r["repair_note"] = fix1.get("repair_note", "")
    r["original_error"] = fail_err.get(rid, fix1.get("spec", {}).get("blocks", [{}])[-1].get("original_error", ""))

    # syntax-check every fixed definition (compile() only; no torch runtime)
    for b in fixed_blocks:
        d = b.get("definition")
        if not d:
            continue
        try:
            compile(d, f"<{rid}>", "exec")
        except SyntaxError as e:
            syntax_failed.append((rid, str(e)))

    # drop the -fix1 sibling (fix now lives in the original)
    by_id.pop(fid, None)
    dropped += 1
    promoted += 1

# rebuild the row list in original order, minus dropped -fix1 rows
out_rows = [r for r in rows if r["id"] in by_id]

with open(OUT, "w") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")

print(f"promoted (fixed in place): {promoted}")
print(f"dropped (-fix1 parking):   {dropped}")
print(f"final row count:           {len(out_rows)}")
print(f"syntax-failed defs:        {len(syntax_failed)}")
for rid, e in syntax_failed:
    print(f"  SYNTAX FAIL {rid}: {e}")
