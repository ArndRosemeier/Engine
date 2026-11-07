#!/usr/bin/env python3
"""Simple launcher UI for Engine examples.

Every folder under examples/ gets a button. VARIANTS adds extra buttons for
examples that take environment variables or arguments (texture families,
the character viewer, close-ups); add a line there for a new variant.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"


@dataclass
class Launch:
    label: str
    package: str
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)


# Extra ways to start an example, shown after the plain ones.
VARIANTS: list[Launch] = [
    Launch("starter (low quality, 60 FPS target)", "starter",
           {"ENGINE_QUALITY": "low", "ENGINE_TARGET_FPS": "60"}),
    Launch("starter (morning)", "starter", {"ENGINE_SKY_HOUR": "8.5"}),
    Launch("starter without grass (compare)", "starter", {"ENGINE_GRASS": "0"}),
    Launch("texgen: town ground", "texgen", {"ENGINE_TEXGEN_FAMILY": "ground"}),
    Launch("texgen: facades", "texgen", {"ENGINE_TEXGEN_FAMILY": "facade"}),
    Launch("texgen: roofs", "texgen", {"ENGINE_TEXGEN_FAMILY": "roof"}),
    Launch("texgen: medieval", "texgen", {"ENGINE_TEXGEN_FAMILY": "medieval"}),
    Launch("vegetation: autumn storm", "vegetation",
           {"ENGINE_SEASON": "0.85", "ENGINE_WIND": "1.8"}),
    Launch("vegetation: rocks close up", "vegetation", {"ENGINE_ROCKS_CLOSE": "1"}),
    Launch("sky_day: sunset", "sky_day", {"ENGINE_SKY_HOUR": "18.8"}),
    Launch("sky_day: storm", "sky_day", {"ENGINE_SKY_WEATHER": "storm"}),
    Launch("lod_forest: LOD off (compare)", "lod_forest", {"ENGINE_LOD": "off"}),
    Launch("model_viewer: Actors character", "model_viewer", {},
           ["examples/model_viewer/assets/character.glb"]),
    Launch("model_viewer: prop pack", "model_viewer", {},
           ["examples/model_viewer/assets/props.fmpk"]),
    Launch("anything without TAA (starter)", "starter", {"ENGINE_TAA": "0"}),
    Launch("starter with 4x MSAA on top of TAA (compare)", "starter", {"ENGINE_MSAA": "4"}),
    Launch("vegetation with 4x MSAA on top of TAA (compare)", "vegetation", {"ENGINE_MSAA": "4"}),
    Launch("water: shore close up", "water", {"ENGINE_WATER_VIEW": "shore"}),
    Launch("water: under the lake", "water", {"ENGINE_WATER_VIEW": "bed"}),
    Launch("water: the stream", "water", {"ENGINE_WATER_VIEW": "river"}),
    Launch("water: swim and dive (G toggles flying)", "water",
           {"ENGINE_WATER_VIEW": "shore", "ENGINE_WATER_BODY": "1"}),
    Launch("water: tropical colour", "water",
           {"ENGINE_WATER_VIEW": "shore", "ENGINE_WATER_LOOK": "tropical"}),
    Launch("coast: on the beach", "coast", {"ENGINE_COAST_VIEW": "beach"}),
    Launch("coast: surf from the water", "coast", {"ENGINE_COAST_VIEW": "surf"}),
    Launch("coast: sea cave", "coast", {"ENGINE_COAST_VIEW": "cave"}),
    Launch("coast: rough sea", "coast", {"ENGINE_COAST_SEA": "rough"}),
    Launch("coast: swim in the surf (G toggles flying)", "coast",
           {"ENGINE_COAST_VIEW": "beach", "ENGINE_WATER_BODY": "1"}),
    Launch("island: from the hill", "island", {"ENGINE_ISLAND_VIEW": "hill"}),
    Launch("island: seen from the sea", "island", {"ENGINE_ISLAND_VIEW": "sea"}),
    Launch("island: evening", "island", {"ENGINE_SKY_HOUR": "19.2"}),
    Launch("lamps: by the fire", "lamps", {"ENGINE_LAMPS_VIEW": "fire"}),
    Launch("lamps: from above", "lamps", {"ENGINE_LAMPS_VIEW": "above"}),
    Launch("lamps: at dusk", "lamps", {"ENGINE_SKY_HOUR": "19.6"}),
    Launch("prefabs: windmill", "prefabs", {"ENGINE_PREFABS_VIEW": "mill"}),
    Launch("prefabs: lantern in hand", "prefabs", {"ENGINE_PREFABS_VIEW": "walker"}),
    Launch("prefabs: carts dropped from the sky", "prefabs", {"ENGINE_PREFABS_DROP": "1"}),
    Launch("characters: from the side", "characters", {"ENGINE_CHARACTERS_VIEW": "side"}),
    Launch("characters: without foot IK (compare)", "characters",
           {"ENGINE_CHARACTERS_VIEW": "side", "ENGINE_CHARACTERS_IK": "0"}),
    Launch("navigation: walkable area shown", "navigation", {"ENGINE_NAV_OVERLAY": "1"}),
    Launch("navigation: stairs to the terrace", "navigation", {"ENGINE_NAV_VIEW": "terrace"}),
    Launch("physics: demo (a throw, dominoes, a blast)", "physics", {"ENGINE_PHYSICS_DEMO": "1"}),
    Launch("buildings: demo (a topple and a pancake)", "buildings", {"ENGINE_BUILDINGS_DEMO": "1"}),
    Launch("fixed_tick: 10 Hz simulation", "fixed_tick", {"ENGINE_SIM_HZ": "10"}),
    Launch("ragdolls: demo (knocks people over by itself)", "ragdolls", {"ENGINE_RAGDOLLS_DEMO": "1"}),
    Launch("interiors: culling off (compare)", "interiors", {"ENGINE_INTERIORS_CULL": "0"}),
    Launch("interiors: no interiors (compare)", "interiors", {"ENGINE_INTERIORS_FURNISH": "0"}),
    Launch("crowds: seen from above", "crowds", {"ENGINE_CROWD_VIEW": "above"}),
    Launch("crowds: 3,000 people", "crowds", {"ENGINE_CROWD_COUNT": "3000"}),
    Launch("fog_volumes: parked in the smog", "fog_volumes", {"ENGINE_FOG_AT": "smog"}),
    Launch("decals: without automatic spraying", "decals", {"ENGINE_DECALS_SPRAY": "0"}),
    Launch("reflections: in the hall", "reflections", {"ENGINE_REFLECTIONS_VIEW": "hall"}),
    Launch("reflections: without SSR (compare)", "reflections", {"ENGINE_REFLECTIONS_SSR": "0"}),
    Launch("starter with ACES tone mapping (compare)", "starter", {"ENGINE_TONEMAP": "aces"}),
    Launch("zfight: the fixed yard", "zfight", {"ENGINE_ZFIGHT_FIXED": "1"}),
    Launch("buildings: find z-fighting (magenta posts)", "buildings", {"ENGINE_ZFIGHT": "1"}),
    Launch("interiors: find z-fighting (magenta posts)", "interiors", {"ENGINE_ZFIGHT": "100"}),
]


def discover_examples() -> list[Launch]:
    if not EXAMPLES_DIR.is_dir():
        raise FileNotFoundError(f"Examples folder not found: {EXAMPLES_DIR}")

    names = [
        entry.name
        for entry in sorted(EXAMPLES_DIR.iterdir(), key=lambda p: p.name.lower())
        if entry.is_dir() and (entry / "Cargo.toml").is_file()
    ]
    known = set(names)
    return [Launch(name, name) for name in names] + [
        v for v in VARIANTS if v.package in known
    ]


def run_example(
    window: tk.Tk,
    launch: Launch,
    status: tk.StringVar,
    buttons: list[tk.Button],
) -> None:
    status.set(f"Running {launch.label}…")
    for button in buttons:
        button.configure(state=tk.DISABLED)

    def on_done(returncode: int) -> None:
        if returncode == 0:
            status.set(f"Finished {launch.label}")
        else:
            status.set(f"{launch.label} exited with code {returncode}")
        for button in buttons:
            button.configure(state=tk.NORMAL)

    # Release builds: the generators and renderer are far slower without
    # optimisation.
    command = ["cargo", "run", "--release", "-p", launch.package]
    if launch.args:
        command += ["--", *launch.args]
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, **launch.env},
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


def build_ui(launches: list[Launch]) -> tk.Tk:
    window = tk.Tk()
    window.title("Engine Examples")
    window.minsize(320, 240)
    window.configure(padx=16, pady=16)

    tk.Label(window, text="Choose an example to run", font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
    )

    status = tk.StringVar(value="Ready")
    buttons: list[tk.Button] = []
    plain = [l for l in launches if not l.env and not l.args]
    variants = [l for l in launches if l.env or l.args]
    for column, (title, group) in enumerate((("Examples", plain), ("Variants", variants))):
        tk.Label(window, text=title, font=("Segoe UI", 10, "bold"), fg="#444").grid(
            row=1, column=column, sticky="w", padx=4
        )
        for row, launch in enumerate(group, start=2):
            button = tk.Button(
                window,
                text=launch.label,
                font=("Segoe UI", 10),
                anchor="w",
                padx=10,
                pady=4,
                command=lambda l=launch: run_example(window, l, status, buttons),
            )
            button.grid(row=row, column=column, sticky="ew", padx=4, pady=2)
            buttons.append(button)

    rows = max(len(plain), len(variants)) + 2
    tk.Label(window, textvariable=status, font=("Segoe UI", 9), fg="#444").grid(
        row=rows, column=0, columnspan=2, sticky="w", pady=(16, 0)
    )
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    return window


def main() -> None:
    launches = discover_examples()
    if not launches:
        raise SystemExit(f"No examples found in {EXAMPLES_DIR}")

    window = build_ui(launches)
    window.mainloop()


if __name__ == "__main__":
    main()
