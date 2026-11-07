#!/usr/bin/env python3
"""Raise the Engine version by 0.001 and add a "What's new" entry.

    python tools/bump_version.py "Short note" ["Another note" ...]
        Adds the next version on top of CHANGELOG.md with today's date.

    python tools/bump_version.py --check [BASE_REF]
        Fails unless this branch's CHANGELOG.md is exactly one version above
        BASE_REF's (default origin/main). CI runs this on every pull request.

    python tools/bump_version.py --renumber [BASE_REF]
        After a rebase where another merge took your number: moves your
        entries on top of BASE_REF's newest version again.

The version lives only in CHANGELOG.md (newest "## <version> (<date>)"
heading first); the engine reads it at build time.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
HEADING = re.compile(r"^## (\d+)\.(\d{3})\b.*$", re.M)


def builds(text: str) -> list[int]:
    return [int(a) * 1000 + int(b) for a, b in HEADING.findall(text)]


def fmt(build: int) -> str:
    return f"{build // 1000}.{build % 1000:03d}"


def base_text(ref: str) -> str:
    """CHANGELOG.md on `ref`, or "" before versioning existed there."""
    r = subprocess.run(
        ["git", "show", f"{ref}:CHANGELOG.md"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0 and "does not exist" not in r.stderr and "exists on disk" not in r.stderr:
        sys.exit(f"git show {ref}:CHANGELOG.md failed: {r.stderr.strip()}")
    return r.stdout if r.returncode == 0 else ""


def newest(text: str) -> int:
    b = builds(text)
    return b[0] if b else 0


def add(notes: list[str]) -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    m = HEADING.search(text)
    if not m:
        sys.exit("CHANGELOG.md has no version entry")
    new = builds(text)[0] + 1
    today = dt.date.today().isoformat()
    body = "\n".join(f"- {n.strip()}" for n in notes)
    entry = f"## {fmt(new)} ({today})\n\n{body}\n\n"
    CHANGELOG.write_text(text[: m.start()] + entry + text[m.start():], encoding="utf-8")
    print(f"Engine version is now {fmt(new)}")


def check(ref: str) -> None:
    mine, base = newest(CHANGELOG.read_text(encoding="utf-8")), newest(base_text(ref))
    want = base + 1
    if mine != want:
        sys.exit(
            f"CHANGELOG.md tops out at {fmt(mine)}, but {ref} is at {fmt(base)}: "
            f"this merge must add exactly one entry, {fmt(want)}. "
            "Run: python tools/bump_version.py \"what this merge brings\" "
            "(or --renumber after a rebase)."
        )
    print(f"OK: {fmt(base)} -> {fmt(want)}")


def renumber(ref: str) -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    base = base_text(ref)
    top = newest(base)
    # This branch's own entries sit above the base's newest heading.
    m = HEADING.search(base)
    cut = text.find(m.group(0)) if m else len(text)
    if cut < 0:
        sys.exit(f"cannot find {ref}'s newest entry in CHANGELOG.md; fix by hand")
    head, tail = text[:cut], text[cut:]
    n = len(HEADING.findall(head))
    if n == 0:
        sys.exit("this branch adds no entry; use the add form instead")
    counter = iter(range(n))

    def repl(h: re.Match) -> str:
        return re.sub(r"\d+\.\d{3}", fmt(top + n - next(counter)), h.group(0), count=1)

    CHANGELOG.write_text(HEADING.sub(repl, head) + tail, encoding="utf-8")
    print(f"Renumbered {n} entr{'y' if n == 1 else 'ies'}; newest is now {fmt(top + n)}")


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
    elif argv[0] == "--check":
        check(argv[1] if len(argv) > 1 else "origin/main")
    elif argv[0] == "--renumber":
        renumber(argv[1] if len(argv) > 1 else "origin/main")
    else:
        add(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
