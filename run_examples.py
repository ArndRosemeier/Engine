#!/usr/bin/env python3
"""Simple launcher UI for Engine examples."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"


def discover_examples() -> list[str]:
    if not EXAMPLES_DIR.is_dir():
        raise FileNotFoundError(f"Examples folder not found: {EXAMPLES_DIR}")

    names: list[str] = []
    for entry in sorted(EXAMPLES_DIR.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir() and (entry / "Cargo.toml").is_file():
            names.append(entry.name)
    return names


def run_example(
    window: tk.Tk,
    name: str,
    status: tk.StringVar,
    buttons: list[tk.Button],
) -> None:
    status.set(f"Running {name}…")
    for button in buttons:
        button.configure(state=tk.DISABLED)

    def on_done(returncode: int) -> None:
        if returncode == 0:
            status.set(f"Finished {name}")
        else:
            status.set(f"{name} exited with code {returncode}")
        for button in buttons:
            button.configure(state=tk.NORMAL)

    try:
        process = subprocess.Popen(
            ["cargo", "run", "-p", name],
            cwd=ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )
    except FileNotFoundError as exc:
        status.set(f"Failed to start cargo: {exc}")
        for button in buttons:
            button.configure(state=tk.NORMAL)
        return

    def poll() -> None:
        code = process.poll()
        if code is None:
            window.after(250, poll)
            return
        on_done(code)

    window.after(250, poll)


def build_ui(examples: list[str]) -> tk.Tk:
    window = tk.Tk()
    window.title("Engine Examples")
    window.minsize(320, 240)
    window.configure(padx=16, pady=16)

    tk.Label(window, text="Choose an example to run", font=("Segoe UI", 12, "bold")).pack(
        anchor="w", pady=(0, 12)
    )

    status = tk.StringVar(value="Ready")
    buttons: list[tk.Button] = []

    for name in examples:
        button = tk.Button(
            window,
            text=name,
            font=("Segoe UI", 11),
            anchor="w",
            padx=12,
            pady=8,
            command=lambda n=name: run_example(window, n, status, buttons),
        )
        button.pack(fill="x", pady=4)
        buttons.append(button)

    tk.Label(window, textvariable=status, font=("Segoe UI", 9), fg="#444").pack(
        anchor="w", pady=(16, 0)
    )
    return window


def main() -> None:
    examples = discover_examples()
    if not examples:
        raise SystemExit(f"No examples found in {EXAMPLES_DIR}")

    window = build_ui(examples)
    window.mainloop()


if __name__ == "__main__":
    main()
