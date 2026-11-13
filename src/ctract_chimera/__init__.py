"""Differentiable CHIMERA geometry, coupled sources, airway and convolution.

Positive-temperature paths differentiate their actual continuous forward laws.
See README.md for hard-limit guarantees, units and approximation boundaries.
"""

from .acoustics import Elements, HumanAcoustics
from .anatomy import ARTIC_NAMES, NEUTRAL, Anatomy
from .coupled import CoupledSynth
from .discretisation import AnatomicalRenderer, AnatomicalSynth, AnatomyDomain
from .geometry import ExactGeometry
from .glottis import DEFAULT_GLOTTIS, GLOTTIS_NAMES
from .renderer import Renderer
from .source import glottal_shape
from .source_random import SourceRandomness
from .topology import build_human
from .turbulence import NoiseInputs

__all__ = [
    "ARTIC_NAMES",
    "DEFAULT_GLOTTIS",
    "GLOTTIS_NAMES",
    "CoupledSynth",
    "SourceRandomness",
    "NEUTRAL",
    "NoiseInputs",
    "Anatomy",
    "AnatomyDomain",
    "AnatomicalRenderer",
    "AnatomicalSynth",
    "Elements",
    "ExactGeometry",
    "HumanAcoustics",
    "Renderer",
    "build_human",
    "glottal_shape",
]
