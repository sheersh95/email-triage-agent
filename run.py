#!/usr/bin/env python3
"""Launcher script — run from project root with `python run.py`.

Sets PYTHONPATH correctly and execs streamlit so `from src.X import Y`
always works, regardless of where the user invokes it from.

Usage:
    python run.py        # starts the Streamlit UI
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "src" / "app.py"


def main() -> None:
    if not APP_PATH.exists():
        sys.exit(f"Can't find {APP_PATH}. Are you in the project root?")

    # Make src/* importable as packages
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    print(f"Launching: {' '.join(cmd)}")
    print(f"PYTHONPATH={env['PYTHONPATH']}\n")

    # exec replaces current process — clean Ctrl-C behavior
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
