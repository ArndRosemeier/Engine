#!/usr/bin/env python3
"""Build an example (or any binary package) and run it as a verify run.

    python tools/verify.py starter
    python tools/verify.py physics --frames 300 --shots 60,120
    python tools/verify.py starter --reference refs/starter.png
    python tools/verify.py starter --save-reference refs/starter.png

The program runs headless-friendly for a fixed number of frames on a fixed
time step, then writes into target/verify/<package>/:

    screenshot.png   the last frame
    shot-<N>.png     extra frames asked for with --shots
    report.json      passed/failed, problems, the game's world.check() results
                     and world.report() values, entity/body/lamp counts, camera

The exit code is 0 when the run passed, 1 when it failed. On Linux without a
display it runs under xvfb-run when that is installed. See engine/src/verify.rs.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", help="cargo package to run, e.g. starter")
    ap.add_argument("--frames", type=int, default=120, help="frames to run (default 120)")
    ap.add_argument("--shots", help="extra screenshot frames, e.g. 30,60")
    ap.add_argument("--reference", help="compare the last frame with this PNG")
    ap.add_argument("--tolerance", type=float, help="share of pixels that may differ (default 0.002)")
    ap.add_argument("--save-reference", help="after a passing run, copy the screenshot here")
    ap.add_argument("--out", help="output folder (default target/verify/<package>)")
    ap.add_argument("--debug", action="store_true", help="debug build instead of release")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    profile = [] if args.debug else ["--release"]
    build = subprocess.run(["cargo", "build", "-p", args.package, *profile], cwd=ROOT)
    if build.returncode != 0:
        print("verify: build failed", file=sys.stderr)
        return 1

    exe = ROOT / "target" / ("debug" if args.debug else "release") / args.package
    if os.name == "nt":
        exe = exe.with_suffix(".exe")
    out = Path(args.out) if args.out else ROOT / "target" / "verify" / args.package
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        report_path.unlink()

    env = dict(os.environ)
    env["ENGINE_VERIFY"] = str(out)
    env["ENGINE_VERIFY_FRAMES"] = str(args.frames)
    env.setdefault("ENGINE_WIDTH", str(args.width))
    env.setdefault("ENGINE_HEIGHT", str(args.height))
    if args.shots:
        env["ENGINE_VERIFY_SHOTS"] = args.shots
    if args.reference:
        env["ENGINE_VERIFY_REFERENCE"] = str(Path(args.reference).resolve())
    if args.tolerance is not None:
        env["ENGINE_VERIFY_TOLERANCE"] = str(args.tolerance)

    cmd = [str(exe)]
    if sys.platform.startswith("linux") and not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        if shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", "-s", f"-screen 0 {args.width}x{args.height}x24", *cmd]
        else:
            print("verify: no display and no xvfb-run; install xvfb", file=sys.stderr)
            return 1
    run = subprocess.run(cmd, cwd=ROOT / "examples" / args.package if (ROOT / "examples" / args.package).is_dir() else ROOT, env=env)

    if not report_path.exists():
        print(f"verify: FAILED, no report written (exit code {run.returncode}); "
              "the program crashed before its first frame or never reached the last one", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text())
    passed = bool(report.get("passed")) and run.returncode == 0
    print(f"verify: {'PASSED' if passed else 'FAILED'}  {args.package}  frames={report.get('frames')}  "
          f"entities={report.get('world', {}).get('entities')}  report={report_path}")
    for problem in report.get("problems", []):
        print(f"  - {problem}")
    if passed and args.save_reference:
        dst = Path(args.save_reference)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "screenshot.png", dst)
        print(f"verify: saved reference {dst}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
