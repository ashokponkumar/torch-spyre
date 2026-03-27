#!/usr/bin/env python3
"""Find example tests for each skip reason, then look up their decorators in source."""
import xml.etree.ElementTree as ET
import os, glob, collections, re

base = "/mnt/devwork/discovery/2026-03-27-discovery/workers"
pytorch = "/mnt/devwork/discovery/2026-03-27-discovery/workspace/pytorch/test"

# Collect examples: {skip_reason: [(file, test_name, classname), ...]}
examples = collections.defaultdict(list)

# Normalize skip reasons into categories
def categorize(msg):
    if not msg:
        return None
    if "onlyNativeDeviceTypes" in msg:
        return "onlyNativeDeviceTypes"
    if "Only runs on ['cpu', 'spyre']" in msg:
        return "Only runs on cpu+spyre"
    if "requires gpu and triton" in msg or msg == "triton":
        return "requires triton"
    if "doesn't support autograd" in msg or "does not support inplace autograd" in msg:
        return "autograd not supported"
    if "Only runs on ['cuda', 'spyre']" in msg:
        return "Only runs on cuda+spyre"
    if "Only runs on cuda" in msg or "Requires CUDA" in msg or "CUDA not found" in msg or "requires cuda" in msg:
        return "requires CUDA"
    if "Scipy required" in msg:
        return "requires scipy"
    if "does support gradgrad" in msg:
        return "gradgrad not supported"
    if "Excluded from CUDA" in msg:
        return "excluded from CUDA tests"
    return None

for d in sorted(glob.glob(f"{base}/*/"), key=lambda x: int(os.path.basename(x.rstrip("/")))):
    idx = int(os.path.basename(d.rstrip("/")))
    xml_path = os.path.join(d, "results.xml")
    test_file_path = os.path.join(d, "test_file.txt")
    if not os.path.exists(xml_path):
        continue
    try:
        fname = open(test_file_path).read().strip()
    except:
        continue

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for tc in root.iter("testcase"):
            skipped = tc.find("skipped")
            if skipped is not None:
                msg = skipped.get("message") or ""
                cat = categorize(msg)
                if cat and len(examples[cat]) < 3:
                    classname = tc.get("classname", "")
                    name = tc.get("name", "")
                    examples[cat].append((fname, classname, name, msg[:150]))
    except:
        pass

# Now for each category, print example and try to find the test function + decorators in source
for cat in ["onlyNativeDeviceTypes", "Only runs on cpu+spyre", "requires triton",
            "autograd not supported", "Only runs on cuda+spyre", "requires CUDA",
            "requires scipy", "gradgrad not supported", "excluded from CUDA tests"]:
    exs = examples.get(cat, [])
    if not exs:
        continue
    print(f"\n{'='*80}")
    print(f"SKIP REASON: {cat}")
    print(f"{'='*80}")

    for fname, classname, testname, msg in exs[:1]:  # just 1 example per category
        # Strip device suffix to get the base test name
        # e.g. test_foo_spyre_float32 -> test_foo
        base_name = re.sub(r'_spyre_\w+$', '', testname)
        base_name = re.sub(r'_spyre$', '', base_name)

        print(f"\n  File:      {fname}")
        print(f"  Class:     {classname}")
        print(f"  Test:      {testname}")
        print(f"  Base name: {base_name}")
        print(f"  Message:   {msg}")

        # Try to find the test function in the source file
        src_file = os.path.join(pytorch, fname)
        if os.path.exists(src_file):
            try:
                source = open(src_file).read()
                # Find the function definition and preceding decorators
                # Look for def <base_name>( with up to 10 preceding decorator lines
                pattern = rf'((?:^\s*@[^\n]+\n){{0,10}})\s*def\s+{re.escape(base_name)}\s*\('
                match = re.search(pattern, source, re.MULTILINE)
                if match:
                    decorators_and_def = match.group(0)
                    # Get some context: the decorators + first line of def
                    lines = decorators_and_def.strip().split('\n')
                    print(f"\n  Source (decorators + def):")
                    for line in lines:
                        print(f"    {line}")
                else:
                    # Try simpler search
                    for i, line in enumerate(source.split('\n')):
                        if f'def {base_name}(' in line:
                            start = max(0, i-8)
                            context = source.split('\n')[start:i+1]
                            # Only show decorator lines
                            print(f"\n  Source (around line {i+1}):")
                            for cl in context:
                                if cl.strip().startswith('@') or cl.strip().startswith('def '):
                                    print(f"    {cl}")
                            break
                    else:
                        print(f"\n  (could not find def {base_name} in source)")
            except Exception as e:
                print(f"\n  (error reading source: {e})")
        else:
            print(f"\n  (source file not found: {src_file})")
