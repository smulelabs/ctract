"""CHIMERA anatomy/human.rs::HumanBody and control/articulators.rs::ARTIC_SPECS.

Anatomy dimensions are independent configuration values, never inferred from pitch
or chosen by sex/age. Coordinates below are template cm; scale is applied by geometry.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Anatomy:
    """HumanBody's four independent scalars (radius/width in unscaled cm).

    Defaults reproduce CHIMERA's default human template. Any positive finite
    values can be supplied; these are configuration, NOT differentiable tensors.
    Changing them rebuilds geometry landmarks and the airway discretisation.
    """

    scale: float = 1.0
    pharynx: float = 1.0
    tongue_radius: float = 2.25
    lip_width: float = 4.6

    def __post_init__(self) -> None:
        for value in (self.scale, self.pharynx, self.tongue_radius, self.lip_width):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("anatomy dimensions must be finite and positive")


ARTIC_NAMES = (
    "jaw",
    "tbX",
    "tbY",
    "ttX",
    "ttY",
    "ttCurl",
    "root",
    "lipAp",
    "lipProt",
    "lipWidth",
    "labiodental",
    "velum",
    "larynx",
    "epilarynx",
    "ventricular",
    "lateral",
    "dorsum",
)
ARTIC_MIN = [-0.1, 1.0, 4.8, 3.5, 5.0, 0.0, -1.0, -0.4, -0.2, 0.0, 0.0, -0.25, -1.6, 0.0, 0.0, 0.0, 0.0]
ARTIC_MAX = [2.6, 4.8, 9.2, 8.0, 10.8, 1.0, 1.0, 2.4, 1.5, 1.0, 1.0, 1.4, 1.6, 1.0, 1.0, 1.0, 1.0]
NEUTRAL = [0.6, 2.9, 6.85, 6.45, 7.3, 0.0, 0.0, 0.7, 0.2, 0.5, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0]
