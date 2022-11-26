from typing import Iterable

import numpy as np
import pytest

from svgdatagrabber.geometry import Point, Line, Segment, Ray


class TestBase:
    tolerance = 1e-10


class TestPoint(TestBase):
    def test_create_point(self):
        point = Point(1, 2)
        assert abs(point.x - 1) < self.tolerance
        assert abs(point.y - 2) < self.tolerance

    @pytest.mark.parametrize("point", [(1, 2), Point(1, 2), 1 + 2j])
    def test_point_aspoint(self, point: Iterable[float]):
        point = Point.aspoint(point)
        assert abs(point.x - 1) < self.tolerance
        assert abs(point.y - 2) < self.tolerance

    def test_point_eq(self):
        point1 = Point(1, 2)
        point2 = Point(1, 2)
        assert point1 == point2

    def test_point_ne(self):
        point1 = Point(1, 2)
        point2 = Point(2, 1)
        assert point1 != point2

    def test_point_distance(self):
        point1 = Point(1, 2)
        point2 = Point(2, 1)
        assert abs(point1.distance(point2) - np.sqrt(2)) < self.tolerance

    @pytest.mark.parametrize(
        "point1, point2, expected",
        [
            (Point(0, 0), Point(1, 1), np.pi / 4),
            (Point(0, 0), Point(-1, 1), 3 * np.pi / 4),
            (Point(0, 0), Point(-1, -1), -3 * np.pi / 4),
            (Point(0, 0), Point(1, -1), -np.pi / 4),
        ],
    )
    def test_point_direction(self, point1: Point, point2: Point, expected: float):
        assert abs(point1.direction(point2) - expected) < self.tolerance

    @pytest.mark.parametrize("point1, point2, expected", [(Point(0, 0), Point(1, 1), np.array([1, 1]))])
    def test_point_vector(self, point1: Point, point2: Point, expected: np.ndarray):
        assert np.allclose(point1.vector(point2), np.array([1, 1]), atol=self.tolerance)


class TestCreateLine(TestBase):
    def test_create_line_from_coefficients(self):
        line = Line(A=1, B=1, C=0)
        assert abs(line.A - 1) < self.tolerance
        assert abs(line.B - 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

        line = Line.fromCoefficients(A=1, B=1, C=0)
        assert abs(line.A - 1) < self.tolerance
        assert abs(line.B - 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

    def test_create_line_from_two_points(self):
        line = Line(start=Point(0, 0), end=Point(1, 1))
        assert abs(line.A + 1) < self.tolerance
        assert abs(line.B - 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

        line = Line.fromTwoPoints(start=Point(0, 0), end=Point(1, 1))
        assert abs(line.A + 1) < self.tolerance
        assert abs(line.B - 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

    def test_create_line_from_point_and_slope(self):
        line = Line(start=Point(0, 0), slope=1)
        assert abs(line.A - 1) < self.tolerance
        assert abs(line.B + 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

        line = Line.fromPointAndSlope(start=Point(0, 0), slope=1)
        assert abs(line.A - 1) < self.tolerance
        assert abs(line.B + 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

    def test_create_line_from_slope_and_intercept(self):
        line = Line(slope=1, intercept=0)
        assert abs(line.A - 1) < self.tolerance
        assert abs(line.B + 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

        line = Line.fromSlopeAndIntercept(slope=1, intercept=0)
        assert abs(line.A - 1) < self.tolerance
        assert abs(line.B + 1) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

    def test_create_line_from_point_and_angle(self):
        line = Line(start=Point(0, 0), angle=np.deg2rad(45))
        assert abs(line.A - np.cos(np.deg2rad(45))) < self.tolerance
        assert abs(line.B + np.sin(np.deg2rad(45))) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

        line = Line.fromPointAndAngle(start=Point(0, 0), angle=np.deg2rad(45))
        assert abs(line.A - np.cos(np.deg2rad(45))) < self.tolerance
        assert abs(line.B + np.sin(np.deg2rad(45))) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

    def test_create_line_from_angle_and_intercept(self):
        line = Line(angle=np.deg2rad(45), intercept=0)
        assert abs(line.A - np.cos(np.deg2rad(45))) < self.tolerance
        assert abs(line.B + np.sin(np.deg2rad(45))) < self.tolerance
        assert abs(line.C - 0) < self.tolerance

        line = Line.fromAngleAndIntercept(angle=np.deg2rad(45), intercept=0)
        assert abs(line.A - np.cos(np.deg2rad(45))) < self.tolerance
        assert abs(line.B + np.sin(np.deg2rad(45))) < self.tolerance
        assert abs(line.C - 0) < self.tolerance


class TestLineMethods(TestBase):
    @pytest.mark.parametrize("line, point, expected", [(Line(A=1, B=1, C=0), Point(1, 1), np.sqrt(2))])
    def test_line_distance(self, line: Line, point: Point, expected: float):
        assert abs(line.distance(point) - expected) < self.tolerance

    @pytest.mark.parametrize("line, y, expected", [(Line(A=1, B=1, C=1), 0, -1)])
    def test_line_getx(self, line: Line, y: float, expected: float):
        assert abs(line.getx(y) - expected) < self.tolerance

    @pytest.mark.parametrize("line, x, expected", [(Line(A=1, B=1, C=1), 0, -1)])
    def test_line_gety(self, line: Line, x: float, expected: float):
        assert abs(line.gety(x) - expected) < self.tolerance

    @pytest.mark.parametrize(
        "line1, line2, expected",
        [(Line(A=1, B=1, C=0), Line(A=1, B=1, C=1), True), (Line(A=1, B=1, C=0), Line(A=-1, B=1, C=0), False)],
    )
    def test_line_is_parallel(self, line1: Line, line2: Line, expected: bool):
        assert line1.isParallel(line2) == expected

    @pytest.mark.parametrize(
        "line1, line2, expected",
        [(Line(A=1, B=1, C=0), Line(A=1, B=1, C=1), False), (Line(A=1, B=1, C=0), Line(A=-1, B=1, C=0), True)],
    )
    def test_line_is_perpendicular(self, line1: Line, line2: Line, expected: bool):
        assert line1.isPerpendicular(line2) == expected

    @pytest.mark.parametrize("line1, line2, expected", [(Line(A=1, B=1, C=-1), Line(A=1, B=-1, C=1), Point(0, 1))])
    def test_line_intersect(self, line1: Line, line2: Line, expected: Point):
        assert line1.intersect(line2) == expected


class TestLineProperties(TestBase):
    @pytest.mark.parametrize("line, expected", [(Line(A=1, B=1, C=0), -1.0), (Line(A=1, B=-1, C=0), 1.0)])
    def test_line_slope(self, line: Line, expected: float):
        assert abs(line.slope - expected) < self.tolerance

    @pytest.mark.parametrize("line, expected", [(Line(A=1, B=1, C=0), -np.pi / 4), (Line(A=-1, B=1, C=0), np.pi / 4)])
    def test_line_angle(self, line: Line, expected: float):
        assert abs(line.angle - expected) < self.tolerance

    @pytest.mark.parametrize("line, expected", [(Line(A=1, B=1, C=0), 0), (Line(A=1, B=1, C=1), -1)])
    def test_line_intercept(self, line: Line, expected: float):
        assert abs(line.intercept - expected) < self.tolerance

    @pytest.mark.parametrize("line, point, expected", [(Line(A=1, B=-1, C=0), Point(0, 1), Line(A=1, B=-1, C=1))])
    def test_line_parallel(self, line: Line, point: Point, expected: Line):
        assert line.parallel(point) == expected

    @pytest.mark.parametrize("line, point, expected", [(Line(A=1, B=-1, C=0), Point(0, 1), Line(A=1, B=1, C=-1))])
    def test_line_perpendicular(self, line: Line, point: Point, expected: Line):
        assert line.perpendicular(point) == expected


class TestLineOperators(TestBase):
    @pytest.mark.parametrize("line1, line2", [(Line(A=1, B=1, C=0), Line(A=-1, B=-1, C=0))])
    def test_line_eq(self, line1: Line, line2: Line):
        assert line1 == line2

    @pytest.mark.parametrize("line1, line2", [(Line(A=-1, B=1, C=0), Line(A=-1, B=1, C=1))])
    def test_line_ne(self, line1: Line, line2: Line):
        assert line1 != line2

    @pytest.mark.parametrize("points, line", [([Point(0, 0), Point(1, 1), Point(2, 2)], Line(A=-1, B=1, C=0))])
    def test_line_contains_point(self, points: Iterable[Point], line: Line):
        for point in points:
            assert point in line


class TestSegment(TestBase):
    @pytest.mark.parametrize("start, end", [(Point(0, 0), Point(1, 1))])
    def test_create_segment(self, start: Point, end: Point):
        segment = Segment(start=start, end=end)
        assert segment.start == start
        assert segment.end == end

    @pytest.mark.parametrize("start, end, length", [(Point(0, 0), Point(1, 1), np.sqrt(2))])
    def test_segment_length(self, start: Point, end: Point, length: float):
        segment = Segment(start=start, end=end)
        assert abs(segment.length - length) < self.tolerance

    @pytest.mark.parametrize("start, end, midpoint", [(Point(0, 0), Point(1, 1), Point(0.5, 0.5))])
    def test_segment_midpoint(self, start: Point, end: Point, midpoint: Point):
        segment = Segment(start=start, end=end)
        assert segment.midpoint == midpoint

    @pytest.mark.parametrize("start, end, expected", [(Point(0, 0), Point(1, 1), np.pi / 4)])
    def test_segment_direction(self, start: Point, end: Point, expected: float):
        segment = Segment(start=start, end=end)
        assert segment.direction == expected

    @pytest.mark.parametrize("start, end", [(Point(0, 0), Point(1, 1))])
    def test_segment_reverse(self, start: Point, end: Point):
        segment = Segment(start=start, end=end)
        assert segment.reverse() == Segment(start=end, end=start)

    @pytest.mark.parametrize(
        "segment1, segment2, expected",
        [
            (Segment(start=Point(0, 0), end=Point(1, 1)), Segment(start=Point(0, 0), end=Point(1, 1)), True),
            (Segment(start=Point(0, 0), end=Point(1, 1)), Segment(start=Point(0, 0), end=Point(2, 2)), False),
        ],
    )
    def test_segment_eq(self, segment1: Segment, segment2: Segment, expected: bool):
        assert (segment1 == segment2) == expected

    @pytest.mark.parametrize(
        "segment1, segment2, expected",
        [
            (Segment(start=Point(0, 0), end=Point(1, 1)), Segment(start=Point(0, 0), end=Point(1, 1)), False),
            (Segment(start=Point(0, 0), end=Point(1, 1)), Segment(start=Point(0, 0), end=Point(2, 2)), True),
        ],
    )
    def test_segment_ne(self, segment1: Segment, segment2: Segment, expected: bool):
        assert (segment1 != segment2) == expected

    @pytest.mark.parametrize(
        "segment, point, expected",
        [
            (Segment(start=Point(0, 0), end=Point(1, 1)), Point(0, 0), True),
            (Segment(start=Point(0, 0), end=Point(1, 1)), Point(0.5, 0.5), True),
            (Segment(start=Point(0, 0), end=Point(1, 1)), Point(1, 1), True),
            (Segment(start=Point(0, 0), end=Point(1, 1)), Point(-1, -1), False),
            (Segment(start=Point(0, 0), end=Point(1, 1)), Point(2, 2), False),
        ],
    )
    def test_segment_contains_point(self, segment: Segment, point: Point, expected: bool):
        assert (point in segment) == expected


class TestRay(TestBase):

    @pytest.mark.parametrize("start, angle", [(Point(0, 0), np.pi / 4)])
    def test_create_ray(self, start: Point, angle: float):
        ray = Ray(start=start, angle=angle)
        assert ray.start == start
        assert abs(ray.angle - angle + np.pi * (ray.angle - angle < 0)) < self.tolerance

    @pytest.mark.parametrize(
        "ray1, ray2, expected",
        [
            (Ray(start=Point(0, 0), end=Point(1, 1)), Ray(start=Point(0, 0), end=Point(1, 1)), True),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Ray(start=Point(0, 0), end=Point(2, 2)), True),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Ray(start=Point(-1, -1), end=Point(2, 2)), False),
        ],
    )
    def test_segment_eq(self, ray1: Ray, ray2: Ray, expected: bool):
        assert (ray1 == ray2) == expected

    @pytest.mark.parametrize(
        "ray1, ray2, expected",
        [
            (Ray(start=Point(0, 0), end=Point(1, 1)), Ray(start=Point(0, 0), end=Point(1, 1)), False),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Ray(start=Point(0, 0), end=Point(2, 2)), False),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Ray(start=Point(-1, -1), end=Point(2, 2)), True),
        ],
    )
    def test_segment_ne(self, ray1: Ray, ray2: Ray, expected: bool):
        assert (ray1 != ray2) == expected

    @pytest.mark.parametrize(
        "ray, point, expected",
        [
            (Ray(start=Point(0, 0), end=Point(1, 1)), Point(0, 0), True),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Point(0.5, 0.5), True),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Point(1, 1), True),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Point(-1, -1), False),
            (Ray(start=Point(0, 0), end=Point(1, 1)), Point(2, 2), True),
        ],
    )
    def test_ray_contains_point(self, ray: Ray, point: Point, expected: bool):
        assert (point in ray) == expected


class TestLine(TestCreateLine, TestLineMethods, TestLineProperties, TestLineOperators, TestSegment, TestRay):
    pass
