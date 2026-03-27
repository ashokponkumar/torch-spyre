#!/usr/bin/env python3
"""Analyze skip reasons from JUnit XML results."""
import xml.etree.ElementTree as ET
import os, glob, collections

base = "/mnt/devwork/discovery/2026-03-27-discovery/workers"
all_skip_reasons = collections.Counter()
per_file_skips = {}

for d in sorted(glob.glob(f"{base}/*/"), key=lambda x: int(os.path.basename(x.rstrip("/")))):
    idx = int(os.path.basename(d.rstrip("/")))
    xml_path = os.path.join(d, "results.xml")
    test_file_path = os.path.join(d, "test_file.txt")

    if not os.path.exists(xml_path):
        continue

    try:
        fname = open(test_file_path).read().strip() if os.path.exists(test_file_path) else f"index-{idx}"
    except:
        fname = f"index-{idx}"

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        file_reasons = collections.Counter()
        for tc in root.iter("testcase"):
            skipped = tc.find("skipped")
            if skipped is not None:
                msg = (skipped.get("message") or "no message")[:120]
                file_reasons[msg] += 1
                all_skip_reasons[msg] += 1

        if file_reasons:
            per_file_skips[fname] = file_reasons
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")

print("=" * 80)
print("TOP 20 SKIP REASONS (across all files)")
print("=" * 80)
for reason, count in all_skip_reasons.most_common(20):
    print(f"  {count:>6}x  {reason}")

print()
print("=" * 80)
print("TOP SKIP REASON PER FILE (files with most skips)")
print("=" * 80)
for fname, reasons in sorted(per_file_skips.items(), key=lambda x: -sum(x[1].values()))[:15]:
    total_skips = sum(reasons.values())
    top_reason, top_count = reasons.most_common(1)[0]
    print(f"\n  {fname} ({total_skips} skips)")
    for reason, count in reasons.most_common(3):
        print(f"    {count:>5}x  {reason}")
