from abc import ABC

from .geometrybase import GeometryBase


class LineBase(GeometryBase, ABC):
    """A base class for line-like geometries."""

    pass


class StraightLineBase(LineBase, ABC):
    """A base class for straight line-like geometries."""

    pass


class CurveLineBase(LineBase, ABC):
    """A base class for curve line-like geometries."""

    pass
