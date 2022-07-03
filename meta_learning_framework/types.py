from typing import List, Union

import numpy as np
import pandas as pd

Instances = Union[np.ndarray, List[List[float]], pd.DataFrame]

Target = Union[int, float]
Targets = Union[np.ndarray, List[Target]]
