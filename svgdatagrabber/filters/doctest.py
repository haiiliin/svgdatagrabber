import doctest


def test_closedpath():
    from . import closedpath

    doctest.testmod(closedpath)


def test_custom():
    from . import custom

    doctest.testmod(custom)


def test_filterbase():
    from . import filterbase

    doctest.testmod(filterbase)


def test_rectangle():
    from . import rectangle

    doctest.testmod(rectangle)


def test_segmentnumber():
    from . import segmentnumber

    doctest.testmod(segmentnumber)


def test_specialline():
    from . import specialline

    doctest.testmod(specialline)


def test_filters():
    test_closedpath()
    test_custom()
    test_filterbase()
    test_rectangle()
    test_segmentnumber()
    test_specialline()
