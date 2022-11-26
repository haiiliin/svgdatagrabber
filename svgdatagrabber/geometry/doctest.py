import doctest


def test_geometrybase():
    import svgdatagrabber.geometry.geometrybase

    doctest.testmod(svgdatagrabber.geometry.geometrybase)


def test_line():
    import svgdatagrabber.geometry.line

    doctest.testmod(svgdatagrabber.geometry.line)


def test_point():
    import svgdatagrabber.geometry.point

    doctest.testmod(svgdatagrabber.geometry.point)


def test_geometry():
    test_geometrybase()
    test_line()
    test_point()
