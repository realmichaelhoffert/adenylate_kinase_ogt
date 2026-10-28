#!/usr/bin/env python
"""
Verify the frozen ROSEATE reference archive, and compare new results to it.

Needs only the standard library (+ numpy for array comparisons), so it runs in
any future environment, including the refactored sklearn-free one.

Command line:
    # integrity: every file present, unchanged, nothing extra
    python reference/verify_reference.py reference/v1_preprint
    python reference/verify_reference.py reference/v1_preprint --quick      # sizes only

    # compare a results file from refactored code (same keys as expected.json)
    python reference/verify_reference.py reference/v1_preprint --results my_results.json [--atol 0.1]

From tests:
    from reference.verify_reference import Reference
    ref = Reference("reference/v1_preprint")
    ref.assert_matches("fig2a.all_sites", {"r2": r2, "rmse": rmse})
    ref.assert_matches("fig2a.legacy_good_cols", {"r2": r2, "rmse": rmse}, subset=True, atol=1e-6)
    ref.assert_array_close(new_preds, "portable/even_enetcv_ind.npz", key="full_predictions", atol=1e-10)
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

DEFAULT_REF = Path(__file__).resolve().parent / "v1_preprint"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def compare(actual, expected, atol=0.0, rtol=0.0, path="", subset=False):
    """Recursively compare JSON-like values. Returns a list of mismatch messages (empty = match).

    Numbers: |a - e| <= atol + rtol * |e|. Strings, bools, None: exact.
    Dicts: every key in `expected` must be in `actual` (extra keys in `actual` are allowed);
           with subset=True, only keys present in `actual` are compared (at every level).
    Lists: same length, elementwise.
    """
    where = path or "<root>"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{where}: expected dict, got {type(actual).__name__}"]
        out = []
        for k, v in expected.items():
            if k not in actual:
                if not subset:
                    out.append(f"{where}.{k}: missing")
            else:
                out += compare(actual[k], v, atol, rtol, f"{path}.{k}" if path else k, subset)
        return out
    if isinstance(expected, (list, tuple)):
        try:
            actual = list(actual)
        except TypeError:
            return [f"{where}: expected list, got {type(actual).__name__}"]
        if len(actual) != len(expected):
            return [f"{where}: length {len(actual)} != expected {len(expected)}"]
        out = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            out += compare(a, e, atol, rtol, f"{where}[{i}]", subset)
            if len(out) > 20:
                return out + [f"{where}: (further mismatches truncated)"]
        return out
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return [] if actual == expected else [f"{where}: {actual!r} != expected {expected!r}"]
    if isinstance(expected, (int, float)):
        try:
            a = float(actual)
        except (TypeError, ValueError):
            return [f"{where}: {actual!r} is not a number"]
        e = float(expected)
        if math.isnan(e):
            return [] if math.isnan(a) else [f"{where}: {a} != expected nan"]
        if abs(a - e) <= atol + rtol * abs(e):
            return []
        return [f"{where}: {a!r} != expected {e!r} (diff {a - e:.3g}, atol {atol}, rtol {rtol})"]
    return [f"{where}: cannot compare type {type(expected).__name__}"]


class Reference:
    def __init__(self, root=DEFAULT_REF):
        self.root = Path(root).resolve()
        if not (self.root / "MANIFEST.tsv").exists():
            raise FileNotFoundError(f"no MANIFEST.tsv in {self.root}")
        with open(self.root / "expected.json") as f:
            self.expected = json.load(f)

    # ---------------------------------------------------------------- files
    def manifest(self):
        with open(self.root / "MANIFEST.tsv", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def path(self, rel):
        return self.root / rel

    def verify_files(self, quick=False, verbose=True):
        """Returns a list of problems: missing, changed (size or sha256), or unlisted files."""
        problems = []
        rows = self.manifest()
        listed = set()
        for i, row in enumerate(rows, 1):
            rel = row["path"]
            listed.add(rel)
            p = self.root / rel
            if not p.is_file():
                problems.append(f"missing: {rel}")
                continue
            if p.stat().st_size != int(row["bytes"]):
                problems.append(f"size changed: {rel}")
                continue
            if not quick and sha256(p) != row["sha256"]:
                problems.append(f"sha256 changed: {rel}")
            if verbose and i % 50 == 0:
                print(f"  checked {i}/{len(rows)}", file=sys.stderr)
        for p in self.root.rglob("*"):
            rel = str(p.relative_to(self.root))
            if p.is_file() and rel != "MANIFEST.tsv" and rel not in listed:
                problems.append(f"unlisted file: {rel}")
        return problems

    # -------------------------------------------------------------- numbers
    def entry(self, key):
        if key not in self.expected:
            raise KeyError(f"{key!r} not in expected.json; available: {sorted(self.expected)}")
        return self.expected[key]

    def value(self, key):
        return self.entry(key)["value"]

    def check(self, key, actual, atol=None, rtol=None, subset=False):
        """Mismatch messages for one expected.json entry; tolerance defaults to the entry's own.
        subset=True compares only the keys present in `actual` (e.g. r2 and rmse without the site list)."""
        e = self.entry(key)
        atol = e.get("atol", 0.0) if atol is None else atol
        rtol = e.get("rtol", 0.0) if rtol is None else rtol
        return compare(actual, e["value"], atol=atol, rtol=rtol, path=key, subset=subset)

    def assert_matches(self, key, actual, atol=None, rtol=None, subset=False):
        problems = self.check(key, actual, atol, rtol, subset)
        if problems:
            raise AssertionError(f"{key} does not match reference:\n  " + "\n  ".join(problems))

    def check_results(self, results, atol=None, rtol=None, keys=None):
        """Compare a dict of results (same keys as expected.json). Keys absent from results are reported."""
        report = {}
        for key in keys or sorted(self.expected):
            if key not in results:
                report[key] = ["not in results"]
            else:
                report[key] = self.check(key, results[key], atol, rtol)
        return report

    # --------------------------------------------------------------- arrays
    def load_array(self, rel, key=None):
        import numpy as np
        p = self.root / rel
        if p.suffix == ".npz":
            with np.load(p) as z:
                return z[key]
        return np.load(p, mmap_mode="r")

    def assert_array_close(self, actual, rel, key=None, atol=0.0, rtol=0.0):
        import numpy as np
        ref = self.load_array(rel, key)
        actual = np.asarray(actual)
        if actual.shape != ref.shape:
            raise AssertionError(f"{rel}{'[' + key + ']' if key else ''}: shape {actual.shape} != {ref.shape}")
        diff = np.abs(actual.astype(np.float64) - ref.astype(np.float64))
        tol = atol + rtol * np.abs(ref.astype(np.float64))
        bad = diff > tol
        if bad.any():
            raise AssertionError(
                f"{rel}{'[' + key + ']' if key else ''}: {int(bad.sum())} of {bad.size} values differ; "
                f"max abs diff {float(diff.max()):.3g} (atol {atol}, rtol {rtol})")
        return float(diff.max()) if diff.size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", nargs="?", type=Path, default=DEFAULT_REF)
    ap.add_argument("--quick", action="store_true", help="check sizes only, skip sha256")
    ap.add_argument("--results", type=Path, help="JSON of new results keyed like expected.json")
    ap.add_argument("--atol", type=float, default=None, help="override every entry's atol")
    ap.add_argument("--rtol", type=float, default=None, help="override every entry's rtol")
    args = ap.parse_args()

    ref = Reference(args.reference)
    status = 0
    if args.results is None:
        print(f"Verifying {ref.root} ({'sizes only' if args.quick else 'sha256'}) ...")
        problems = ref.verify_files(quick=args.quick)
        for p in problems:
            print("  " + p)
        print("OK: archive intact" if not problems else f"FAILED: {len(problems)} problem(s)")
        status = 1 if problems else 0
        checks = json.loads((ref.root / "checks.json").read_text())
        n_fail = sum(c["status"] == "FAIL" for c in checks)
        if n_fail:
            print(f"Note: {n_fail} build-time consistency check(s) failed; see checks.json")
    else:
        with open(args.results) as f:
            results = json.load(f)
        report = ref.check_results(results, args.atol, args.rtol)
        for key, problems in report.items():
            state = "MISSING" if problems == ["not in results"] else ("OK" if not problems else "DIFF")
            print(f"[{state:7}] {key}")
            if state == "DIFF":
                for p in problems:
                    print("           " + p)
        status = 1 if any(p and p != ["not in results"] for p in report.values()) else 0
    sys.exit(status)


if __name__ == "__main__":
    main()
