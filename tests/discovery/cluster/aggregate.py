#!/usr/bin/env python3
"""Aggregate discovery run results from PVC worker directories."""
import json, os, re, glob

base = "/mnt/devwork/discovery/2026-03-27-discovery/workers"
gtotal = gpass = gfail = gerr = gskip = 0
files_done = 0
rows = []

for d in sorted(glob.glob(f"{base}/*/"), key=lambda x: int(os.path.basename(x.rstrip("/")))):
    idx = int(os.path.basename(d.rstrip("/")))
    sf = os.path.join(d, "summary.json")
    dm = os.path.join(d, "done.marker")
    if not os.path.exists(dm):
        continue
    files_done += 1
    try:
        raw = open(sf).read()
        # Fix duration bug (two floats concatenated like 177.3610.0)
        raw = re.sub(r'("duration": \d+\.\d+)\d+\.\d+', r'\1', raw)
        s = json.loads(raw)
        t = s.get("total", 0)
        p = s.get("passed", 0)
        f = s.get("failed", 0)
        e = s.get("errors", 0)
        sk = s.get("skipped", 0)
        ex = s.get("exit_code", "?")
        fn = s.get("file", "?")
        gtotal += t; gpass += p; gfail += f; gerr += e; gskip += sk
        rows.append((idx, fn, ex, t, p, f, e, sk))
    except Exception as exc:
        rows.append((idx, f"PARSE_ERROR: {exc}", "?", 0, 0, 0, 0, 0))

print(f"{'IDX':>3} {'FILE':<45} {'EXIT':>4} {'TOTAL':>7} {'PASS':>7} {'FAIL':>7} {'ERR':>5} {'SKIP':>6} {'RATE':>5}")
print("-" * 100)
for idx, fn, ex, t, p, f, e, sk in rows:
    pct = f"{p/t*100:.0f}%" if t > 0 else "--"
    print(f"{idx:>3} {fn:<45} {ex:>4} {t:>7} {p:>7} {f:>7} {e:>5} {sk:>6}  {pct}")

print("-" * 100)
print(f"{'TOTAL':<49} {'':>4} {gtotal:>7} {gpass:>7} {gfail:>7} {gerr:>5} {gskip:>6}")
print()
print(f"Files completed:       {files_done}/63")
print(f"Total tests:           {gtotal:,}")
print(f"Passed:                {gpass:,}")
print(f"Failed:                {gfail:,}")
print(f"Errors:                {gerr:,}")
print(f"Skipped:               {gskip:,}")
effective = gtotal - gskip
print(f"Effective (excl skip): {effective:,}")
if effective > 0:
    print(f"Pass rate:             {gpass/effective*100:.1f}%")
    print(f"Fail rate:             {gfail/effective*100:.1f}%")
