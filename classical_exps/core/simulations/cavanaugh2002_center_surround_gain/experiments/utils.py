"""Experiment stimulus generation utilities.

This package contains stimulus generation + experiment runner functions
"""

from __future__ import annotations

# Useful
import os
import time

import numpy as np
import torch
from tqdm import tqdm

# Utils (project)
from classical_exps.core.tools.utils import *  # noqa: F401,F403
from classical_exps.core.tools.utils import SingleCellModel  # noqa: F401

# Image generation
import imagen
from imagen.image import BoundingBox

# Data / optimization / plotting
import h5py
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def fmt_pct(x: float) -> str:
    return f"{100*x:.0f}pct"
