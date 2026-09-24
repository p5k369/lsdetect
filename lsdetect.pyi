import numpy as np
import numpy.typing as npt

class Segment:
    """One detected line segment, in input-image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    precision: float
    log_nfa: float
    @property
    def angle(self) -> float: ...
    @property
    def length(self) -> float: ...

def detect(
    gray: npt.NDArray[np.float64], scale: float = 0.8
) -> list[Segment]: ...
def warp_rgb(
    src: npt.NDArray[np.uint8],
    inverse: tuple[
        float, float, float, float, float, float, float, float, float
    ],
    scale: float,
    off_x: float,
    off_y: float,
) -> npt.NDArray[np.uint8]: ...
