class GeometryBase:
    #: Tolerance for equality.
    tolerance = 1e-6

    def __eq__(self, other: "GeometryBase") -> bool:
        raise NotImplementedError

    def __ne__(self, other: "GeometryBase") -> bool:
        return not self.__eq__(other)
