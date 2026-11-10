#!/usr/bin/env python3
"""Launcher UI for Engine examples.

Every folder under examples/ is listed (searchable, and it scrolls however
many there are). Picking one shows its description, taken from the `//!`
comment at the top of its main.rs, and a button for each way to start it.
VARIANTS adds those extra buttons for examples that take environment
variables or arguments; add a line there for a new variant, labelled
"<example>: what it does".
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


# Extra ways to start an example, shown as buttons next to its "Run".
VARIANTS: list[Launch] = [
    Launch("starter: low quality, 60 FPS target", "starter",
           {"ENGINE_QUALITY": "low", "ENGINE_TARGET_FPS": "60"}),
    Launch("starter: morning", "starter", {"ENGINE_SKY_HOUR": "8.5"}),
    Launch("gameplay: walk the demo route", "gameplay", {"ENGINE_GAMEPLAY_DEMO": "1"}),
    Launch("starter: without grass (compare)", "starter", {"ENGINE_GRASS": "0"}),
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
    Launch("starter: without TAA (compare)", "starter", {"ENGINE_TAA": "0"}),
    Launch("starter: 4x MSAA on top of TAA (compare)", "starter", {"ENGINE_MSAA": "4"}),
    Launch("vegetation: 4x MSAA on top of TAA (compare)", "vegetation", {"ENGINE_MSAA": "4"}),
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
    Launch("island: from high above", "island", {"ENGINE_ISLAND_VIEW": "above"}),
    Launch("island: evening", "island", {"ENGINE_SKY_HOUR": "19.2"}),
    Launch("open_world: flies itself (120 m/s)", "open_world",
           {"ENGINE_AUTOFLY": "1", "ENGINE_OPEN_WORLD_SPEED": "120"}),
    Launch("open_world: evening", "open_world", {"ENGINE_SKY_HOUR": "19.0"}),
    Launch("open_world: another world", "open_world", {"ENGINE_OPEN_WORLD_SEED": "11"}),
    Launch("audio: in the cave", "audio", {"ENGINE_AUDIO_VIEW": "cave"}),
    Launch("audio: in the hall", "audio", {"ENGINE_AUDIO_VIEW": "hall"}),
    Launch("particles: by the fire at dusk", "particles",
           {"ENGINE_PARTICLES_VIEW": "fire", "ENGINE_SKY_HOUR": "19.6"}),
    Launch("lamps: by the fire", "lamps", {"ENGINE_LAMPS_VIEW": "fire"}),
    Launch("lamps: from above", "lamps", {"ENGINE_LAMPS_VIEW": "above"}),
    Launch("lamps: at dusk", "lamps", {"ENGINE_SKY_HOUR": "19.6"}),
    Launch("prefabs: windmill", "prefabs", {"ENGINE_PREFABS_VIEW": "mill"}),
    Launch("prefabs: lantern in hand", "prefabs", {"ENGINE_PREFABS_VIEW": "walker"}),
    Launch("prefabs: carts dropped from the sky", "prefabs", {"ENGINE_PREFABS_DROP": "1"}),
    Launch("anim_states: the scripted demo", "anim_states", {"ENGINE_ANIM_DEMO": "1"}),
    Launch("glass: from the far side", "glass", {"ENGINE_GLASS_YAW": "270"}),
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
    Launch("moons: the one usual moon (compare)", "moons", {"ENGINE_MOONS_ONE": "1"}),
    Launch("fog_volumes: parked in the smog", "fog_volumes", {"ENGINE_FOG_AT": "smog"}),
    Launch("decals: without automatic spraying", "decals", {"ENGINE_DECALS_SPRAY": "0"}),
    Launch("reflections: in the hall", "reflections", {"ENGINE_REFLECTIONS_VIEW": "hall"}),
    Launch("reflections: without SSR (compare)", "reflections", {"ENGINE_REFLECTIONS_SSR": "0"}),
    Launch("starter: ACES tone mapping (compare)", "starter", {"ENGINE_TONEMAP": "aces"}),
    Launch("zfight: the fixed yard", "zfight", {"ENGINE_ZFIGHT_FIXED": "1"}),
    Launch("buildings: find z-fighting (magenta posts)", "buildings", {"ENGINE_ZFIGHT": "1"}),
    Launch("interiors: find z-fighting (magenta posts)", "interiors", {"ENGINE_ZFIGHT": "100"}),
]


def discover_examples() -> list[str]:
    if not EXAMPLES_DIR.is_dir():
        raise FileNotFoundError(f"Examples folder not found: {EXAMPLES_DIR}")
    return [
        entry.name
        for entry in sorted(EXAMPLES_DIR.iterdir(), key=lambda p: p.name.lower())
        if entry.is_dir() and (entry / "Cargo.toml").is_file()
    ]


def describe(name: str) -> str:
    """The example's doc comment (the `//!` lines at the top of main.rs)."""
    main_rs = EXAMPLES_DIR / name / "src" / "main.rs"
    try:
        lines = main_rs.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    doc: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//!"):
            text = stripped[3:]
            doc.append(text[1:] if text.startswith(" ") else text)
        elif doc or stripped:
            break
    return "\n".join(doc).strip()


def launches_for(name: str) -> list[Launch]:
    """The plain launch first, then the example's variants."""
    return [Launch("Run", name)] + [v for v in VARIANTS if v.package == name]


def variant_title(launch: Launch) -> str:
    # Variant labels start with "<example>: "; the example is already shown.
    prefix = f"{launch.package}: "
    return launch.label[len(prefix):] if launch.label.startswith(prefix) else launch.label


class Launcher:
    """A searchable list of examples on the left; on the right, what the
    selected one does and a button for each way to start it."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.docs = {n: describe(n) for n in names}

        self.window = tk.Tk()
        self.window.title("Engine Examples")
        self.window.geometry("1080x680")
        self.window.minsize(640, 360)

        self.status = tk.StringVar(value="Ready")
        tk.Label(
            self.window, textvariable=self.status, font=("Segoe UI", 9), fg="#444", anchor="w"
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))

        left = tk.Frame(self.window, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left, text="Examples", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.query = tk.StringVar()
        search = tk.Entry(left, textvariable=self.query, font=("Segoe UI", 10))
        search.pack(fill=tk.X, pady=(6, 6))
        search.focus_set()
        tk.Label(
            left,
            text="Type to filter, Enter or double-click to run",
            font=("Segoe UI", 8),
            fg="#666",
        ).pack(side=tk.BOTTOM, anchor="w", pady=(6, 0))
        list_frame = tk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 11),
            width=26,
            activestyle="none",
            exportselection=False,
            yscrollcommand=scroll.set,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.configure(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", lambda _: self.show())
        self.listbox.bind("<Double-Button-1>", lambda _: self.run_selected())
        self.listbox.bind("<Return>", lambda _: self.run_selected())
        search.bind("<Return>", lambda _: self.run_selected())
        search.bind("<Down>", lambda _: self.move(1))
        search.bind("<Up>", lambda _: self.move(-1))

        right = tk.Frame(self.window, padx=14, pady=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.title = tk.Label(right, text="", font=("Segoe UI", 14, "bold"))
        self.title.pack(anchor="w")
        self.buttons = tk.Frame(right)
        self.buttons.pack(fill=tk.X, pady=(8, 8))
        text_frame = tk.Frame(right)
        text_frame.pack(fill=tk.BOTH, expand=True)
        text_scroll = tk.Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text = tk.Text(
            text_frame,
            wrap="word",
            font=("Segoe UI", 10),
            relief="flat",
            padx=4,
            background=self.window.cget("background"),
            yscrollcommand=text_scroll.set,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.configure(command=self.text.yview)

        self.shown: list[str] = []
        self.query.trace_add("write", lambda *_: self.refill())
        self.refill()

    def refill(self) -> None:
        words = self.query.get().lower().split()
        selected = self.selected()

        def hit(name: str) -> bool:
            variants = " ".join(v.label for v in VARIANTS if v.package == name)
            hay = f"{name} {self.docs[name]} {variants}".lower()
            return all(w in hay for w in words)

        self.shown = [n for n in self.names if hit(n)]
        self.listbox.delete(0, tk.END)
        for name in self.shown:
            extra = len(launches_for(name)) - 1
            self.listbox.insert(tk.END, f"{name}   +{extra}" if extra else name)
        if self.shown:
            self.select(self.shown.index(selected) if selected in self.shown else 0)
        else:
            self.show()

    def selected(self) -> str | None:
        sel = self.listbox.curselection()
        return self.shown[sel[0]] if sel and sel[0] < len(self.shown) else None

    def select(self, index: int) -> None:
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.show()

    def move(self, step: int) -> str:
        if self.shown:
            sel = self.listbox.curselection()
            index = (sel[0] if sel else 0) + step
            self.select(max(0, min(len(self.shown) - 1, index)))
        return "break"

    def show(self) -> None:
        name = self.selected()
        for child in self.buttons.winfo_children():
            child.destroy()
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        if name is None:
            self.title.configure(text="No example matches")
            self.text.configure(state=tk.DISABLED)
            return
        self.title.configure(text=name)
        # Buttons wrap into rows of three, so any number of variants fits.
        columns = 3
        for i, launch in enumerate(launches_for(name)):
            plain = not launch.env and not launch.args
            tk.Button(
                self.buttons,
                text="Run" if plain else variant_title(launch),
                font=("Segoe UI", 10, "bold" if plain else "normal"),
                anchor="w",
                padx=10,
                pady=3,
                command=lambda l=launch: self.run(l),
            ).grid(row=i // columns, column=i % columns, sticky="ew", padx=3, pady=2)
        for c in range(columns):
            self.buttons.columnconfigure(c, weight=1, uniform="variant")
        self.text.insert("1.0", self.docs[name] or "(no description)")
        self.text.configure(state=tk.DISABLED)

    def run_selected(self) -> str:
        name = self.selected()
        if name is not None:
            self.run(launches_for(name)[0])
        return "break"

    def run(self, launch: Launch) -> None:
        label = launch.package if launch.label == "Run" else launch.label
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
            self.status.set(f"Failed to start cargo: {exc}")
            return
        self.status.set(f"Running {label}…")

        def poll() -> None:
            code = process.poll()
            if code is None:
                self.window.after(250, poll)
            elif code == 0:
                self.status.set(f"Finished {label}")
            else:
                self.status.set(f"{label} exited with code {code}")

        self.window.after(250, poll)


def main() -> None:
    names = discover_examples()
    if not names:
        raise SystemExit(f"No examples found in {EXAMPLES_DIR}")
    Launcher(names).window.mainloop()


if __name__ == "__main__":
    main()
