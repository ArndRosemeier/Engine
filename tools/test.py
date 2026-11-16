#!/usr/bin/env python3
"""Run Engine's tests: the essential ones on every build, all of them on demand.

    python tools/test.py           essential tests only (every build, CI, before a push)
    python tools/test.py --full    every test, including the slow ones (only when asked for)

Essential tests catch bugs that break games built on the engine. Each one
has a budget of 10 ms. This script
measures every essential test on its own and fails when one takes longer,
unless the line right above its `#[test]` reads `// Over budget: <reason>`.

Every other test is marked `#[ignore = "full"]` and runs only with --full,
which has no time budget, together with the doc examples (README, AGENTS.md
and the `///` comments, about 1.5 minutes to compile). A new test is full
unless it guards against a game-breaking bug and fits the budget.

The verify runs of the examples (`python tools/verify.py <example>`) are on
demand too; CI runs them, and --full, only when started by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET_MS = 10.0
LINE = re.compile(r"^test (\S+) \.\.\. (\w+)(?: <([0-9.]+)s>)?")


def test_binaries() -> list[str]:
    build = subprocess.run(
        ["cargo", "test", "-p", "engine", "--lib", "--no-run", "--message-format=json"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    if build.returncode != 0:
        print("test: build failed", file=sys.stderr)
        sys.exit(1)
    bins = []
    for line in build.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("executable") and msg.get("profile", {}).get("test"):
            bins.append(msg["executable"])
    return bins


def reason_over_budget(name: str) -> str | None:
    """The `// Over budget: ...` comment above the test's #[test], if any."""
    fn = name.rsplit("::", 1)[-1]
    pattern = re.compile(rf"^\s*fn {re.escape(fn)}\s*\(")
    for path in (ROOT / "engine" / "src").rglob("*.rs"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not pattern.match(line):
                continue
            j = i - 1
            while j >= 0 and lines[j].strip().startswith("#["):
                j -= 1
            if j >= 0 and lines[j].strip().lower().startswith("// over budget:"):
                return lines[j].strip()[len("// over budget:"):].strip()
    return None


def time_alone(exe: str, name: str, env: dict) -> float:
    """A test run on its own, three times; the fastest run counts (machines are noisy)."""
    best = float("inf")
    for _ in range(3):
        cmd = [exe, "-Z", "unstable-options", "--report-time", "--exact", name]
        run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in run.stdout.splitlines():
            m = LINE.match(line)
            if m and m.group(3):
                best = min(best, float(m.group(3)) * 1000.0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="run every test, not only the essential ones")
    ap.add_argument("filter", nargs="?", help="only tests whose name contains this")
    args = ap.parse_args()

    env = dict(os.environ)
    # Lets the stable test runner time each test (--report-time).
    env["RUSTC_BOOTSTRAP"] = "1"
    results = []  # (name, outcome, ms, exe)
    for exe in test_binaries():
        cmd = [exe, "-Z", "unstable-options", "--report-time"]
        if args.full:
            cmd.append("--include-ignored")
        else:
            # One at a time, so each essential test is timed on its own.
            cmd.append("--test-threads=1")
        if args.filter:
            cmd.append(args.filter)
        run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in run.stdout.splitlines():
            m = LINE.match(line)
            if m:
                results.append((m.group(1), m.group(2), float(m.group(3) or 0) * 1000.0, exe))
        if run.returncode != 0:
            print(run.stdout)

    if not args.full:
        # A test over budget is timed again on its own before it counts as slow.
        results = [
            (n, o, time_alone(exe, n, env) if o == "ok" and ms > BUDGET_MS else ms, exe) for n, o, ms, exe in results
        ]
    if args.full and not args.filter:
        doc = subprocess.run(["cargo", "test", "-p", "engine", "--doc"], cwd=ROOT)
        if doc.returncode != 0:
            results.append(("doc examples", "FAILED", 0.0, ""))
    failed = [r for r in results if r[1] == "FAILED"]
    ran = [r for r in results if r[1] in ("ok", "FAILED")]
    skipped = sum(1 for r in results if r[1] == "ignored")
    over = []
    if not args.full:
        for name, _, ms, _ in ran:
            if ms > BUDGET_MS:
                reason = reason_over_budget(name)
                over.append((name, ms, reason))

    for name, _, ms, _ in sorted(ran, key=lambda r: -r[2])[:5]:
        print(f"  {ms:9.1f} ms  {name}")
    kind = "" if args.full else " essential"
    total = sum(r[2] for r in ran)
    print(f"test: {len(ran)}{kind} tests in {total:.0f} ms, {len(failed)} failed", end="")
    print(f", {skipped} full tests not run (python tools/test.py --full)" if skipped else "")
    bad = [o for o in over if o[2] is None]
    for name, ms, reason in over:
        if reason:
            print(f"test: {name} took {ms:.1f} ms (over budget: {reason})")
        else:
            print(
                f"test: {name} took {ms:.1f} ms, over the {BUDGET_MS:.0f} ms budget. Make it faster, "
                f'mark it #[ignore = "full"], or write `// Over budget: <reason>` above its #[test].',
                file=sys.stderr,
            )
    for name, _, _, _ in failed:
        print(f"test: FAILED {name}", file=sys.stderr)
    return 1 if failed or bad or not ran else 0


if __name__ == "__main__":
    sys.exit(main())
