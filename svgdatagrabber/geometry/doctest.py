import doctest


def test_geometrybase():
    from . import geometrybase

    doctest.testmod(geometrybase)


def test_line():
    from . import line

    doctest.testmod(line)


def test_point():
    from . import point

    doctest.testmod(point)


def test_geometry():
    test_geometrybase()
    test_line()
    test_point()
