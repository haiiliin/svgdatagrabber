from svgdatagrabber.parser import SvgPathParser


def test_getpaths():
    pathPoints = SvgPathParser(
        "praai2013.svg",
        xrange=(0, 3000),
        yrange=(4800, 6000),
        min_segments=4,
        tolerance=1,
    ).parse().lines()
    assert len(pathPoints) == 12
