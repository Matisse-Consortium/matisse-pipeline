# src/matisse_pipeline/types.py

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]
ComplexArray = NDArray[np.complexfloating[Any, Any]]
