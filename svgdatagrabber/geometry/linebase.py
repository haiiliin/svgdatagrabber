from abc import ABC

from .geometrybase import GeometryBase


class LineBase(GeometryBase, ABC):
    """Base class for line-like geometries."""

    pass


class StraightLineBase(LineBase, ABC):
    """Base class for straight line-like geometries."""

    pass


class CurveLineBase(LineBase, ABC):
    """Base class for curve line-like geometries."""

    pass
