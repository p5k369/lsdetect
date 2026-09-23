import numpy as np
import numpy.typing as npt

class Segment:
    """One detected line segment, in input-image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    log_nfa: float
    @property
    def angle(self) -> float: ...
    @property
    def length(self) -> float: ...

def detect(
    gray: npt.NDArray[np.float64], scale: float = 0.8
) -> list[Segment]: ...
