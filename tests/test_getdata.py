from svgdatagrabber.parser import SvgPathParser


def test_getpaths():
    arrays = (
        SvgPathParser(
            "praai2013.svg",
            xrange=(0, 3000),
            yrange=(4800, 6000),
            min_segments=4,
            tolerance=1,
        )
        .parse()
        .arrays
    )
    assert len(arrays) == 12
