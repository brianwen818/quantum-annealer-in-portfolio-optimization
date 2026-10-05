"""設定檔讀取與 repo 相對路徑。"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def set_seed(seed: int) -> None:
    """固定 Python / NumPy（以及已載入時的 TensorFlow）的亂數種子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
