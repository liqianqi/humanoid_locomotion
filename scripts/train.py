"""Register RSX tasks, then run Isaac Lab's RSL-RL trainer.

Does not modify Isaac Lab. Usage:

    /home/ubuntu/IsaacLab/isaaclab.sh -p /home/ubuntu/humanoid_locomotion/scripts/train.py \\
      --task Isaac-Velocity-Flat-RSX-v0 --num_envs 16
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import humanoid_locomotion  # noqa: F401

_TRAIN_CANDIDATES = [
    Path("/home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"),
    Path(__file__).resolve().parents[2].parent / "IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py",
]


def _official_train_script() -> Path:
    for path in _TRAIN_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find Isaac Lab rsl_rl/train.py")


if __name__ == "__main__":
    train_py = _official_train_script()
    sys.path.insert(0, str(train_py.parent))
    runpy.run_path(str(train_py), run_name="__main__")
