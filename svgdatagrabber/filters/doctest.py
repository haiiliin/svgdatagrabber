import doctest


def test_closedpath():
    import svgdatagrabber.filters.closedpath

    doctest.testmod(svgdatagrabber.filters.closedpath)


def test_custom():
    import svgdatagrabber.filters.custom

    doctest.testmod(svgdatagrabber.filters.custom)


def test_filterbase():
    import svgdatagrabber.filters.filterbase

    doctest.testmod(svgdatagrabber.filters.filterbase)


def test_rectangle():
    import svgdatagrabber.filters.rectangle

    doctest.testmod(svgdatagrabber.filters.rectangle)


def test_segmentnumber():
    import svgdatagrabber.filters.segmentnumber

    doctest.testmod(svgdatagrabber.filters.segmentnumber)


def test_specialline():
    import svgdatagrabber.filters.specialline

    doctest.testmod(svgdatagrabber.filters.specialline)


def test_filters():
    test_closedpath()
    test_custom()
    test_filterbase()
    test_rectangle()
    test_segmentnumber()
    test_specialline()
