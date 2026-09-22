"""Register RSX tasks, then run Isaac Lab's RSL-RL play script."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import humanoid_locomotion  # noqa: F401

_PLAY_CANDIDATES = [
    Path("/home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"),
    Path(__file__).resolve().parents[2].parent / "IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py",
]


def _official_script() -> Path:
    for path in _PLAY_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find Isaac Lab rsl_rl/play.py")


if __name__ == "__main__":
    # anchor relative "logs/..." lookups and Hydra's "outputs/" to the project root
    os.chdir(Path(__file__).resolve().parents[1])
    play_py = _official_script()
    sys.path.insert(0, str(play_py.parent))
    runpy.run_path(str(play_py), run_name="__main__")
