#!/usr/bin/env python3
"""Start a new example (or game) crate with every list entry filled in.

    python tools/new_example.py gem_hunt "Collect five gems while an enemy patrols"
    python tools/new_example.py gem_hunt "..." --find "Collecting things, a patrolling enemy"
    python tools/new_example.py gem_hunt "..." --frames 900 --no-ci

Creates examples/<name>/ (Cargo.toml and a src/main.rs that starts from the
game skeleton in AGENTS.md, which compiles) and adds the crate to:
  - the workspace members in Cargo.toml
  - the Examples table in README.md
  - a verify step in .github/workflows/ci.yml (unless --no-ci)
  - the "Where to find things" table in AGENTS.md (only with --find)
Then build and check it with `python tools/verify.py <name>`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARGO = """[package]
name = "{name}"
version.workspace = true
edition.workspace = true
license.workspace = true
publish = false

[lints]
workspace = true

[dependencies]
engine = {{ workspace = true }}
"""


def skeleton(title: str, about: str) -> str:
    """The AGENTS.md game skeleton, with this example's title and vsync off."""
    guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    start = guide.index("## A game skeleton")
    code = guide[guide.index("```rust,no_run\n", start) + len("```rust,no_run\n"):]
    code = code[: code.index("```")]
    code = code.replace('"my game"', f'"{title}"')
    code = code.replace(".quality(Quality::High);", ".quality(Quality::High).vsync(false);")
    return f"//! {about}\n" + code


def insert_after_last(text: str, pattern: str, line: str, where: str) -> str:
    matches = list(re.finditer(pattern, text, re.M))
    if not matches:
        sys.exit(f"new_example: could not find {where}")
    end = text.index("\n", matches[-1].end()) + 1
    return text[:end] + line + text[end:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="crate name: lowercase letters, digits and _")
    ap.add_argument("about", help="one line on what it shows (README table, top of main.rs)")
    ap.add_argument("--find", help='a "You want" line for the AGENTS.md table')
    ap.add_argument("--frames", type=int, default=600, help="frames for the CI verify step")
    ap.add_argument("--no-ci", action="store_true", help="no CI verify step")
    args = ap.parse_args()

    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        sys.exit("new_example: the name must be lowercase letters, digits and _, starting with a letter")
    folder = ROOT / "examples" / name
    if folder.exists():
        sys.exit(f"new_example: {folder} already exists")

    (folder / "src").mkdir(parents=True)
    (folder / "Cargo.toml").write_text(CARGO.format(name=name), encoding="utf-8")
    (folder / "src" / "main.rs").write_text(skeleton(name.replace("_", " "), args.about), encoding="utf-8")
    done = [f"examples/{name}/ (Cargo.toml, src/main.rs from the AGENTS.md skeleton)"]

    cargo = ROOT / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    text = insert_after_last(text, r'^    "examples/[^"]+",$', f'    "examples/{name}",\n', "workspace members")
    cargo.write_text(text, encoding="utf-8")
    done.append("Cargo.toml workspace members")

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    examples = text.index("\n## Examples")
    after = text.index("\n## ", examples + 1)
    table = insert_after_last(text[examples:after], r"^\| `[a-z0-9_]+` \|.*$", f"| `{name}` | {args.about} |\n", "the README Examples table")
    readme.write_text(text[:examples] + table + text[after:], encoding="utf-8")
    done.append("README.md Examples table")

    if not args.no_ci:
        ci = ROOT / ".github" / "workflows" / "ci.yml"
        text = ci.read_text(encoding="utf-8")
        step = (
            f"      - name: {name} ({args.about[:60]})\n"
            f"        run: python3 tools/verify.py {name} --frames {args.frames} --width 640 --height 360\n\n"
        )
        mark = "      - name: Keep the reports and screenshots"
        if mark not in text:
            sys.exit("new_example: could not find where the verify steps end in ci.yml")
        ci.write_text(text.replace(mark, step + mark, 1), encoding="utf-8")
        done.append(f".github/workflows/ci.yml verify step ({args.frames} frames)")

    if args.find:
        agents = ROOT / "AGENTS.md"
        text = agents.read_text(encoding="utf-8")
        text = insert_after_last(text, r"^\| .* \| .*`examples/[a-z0-9_]+`.* \|$", f"| {args.find} | `examples/{name}` |\n", "the AGENTS.md table")
        agents.write_text(text, encoding="utf-8")
        done.append('AGENTS.md "Where to find things" table')

    print("new_example: made " + name)
    for d in done:
        print("  - " + d)
    print(f"next: edit examples/{name}/src/main.rs, then `python tools/verify.py {name}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
