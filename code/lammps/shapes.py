from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import pi

BOX_SIZE = 25


def binary_search(f, a, b, tol=1e-6):
    """
    Performs a binary search for the root of a function f between a and b
    :param f: The function to find the root of
    :param a: The lower bound
    :param b: The upper bound
    :param tol: The tolerance
    :return: The root
    """
    while abs(b - a) > tol:
        c = (a + b) / 2
        f_c = f(c)
        if f_c == 0:
            return c
        elif f_c > 0:
            b = c
        else:
            a = c
    return (a + b) / 2


def generate_bcc_lattice_points(lattice_size=BOX_SIZE, lattice_spacing=2.8665, offset=(0, 0, 0)):
    s = [*reversed([-x for x in np.arange(lattice_spacing, lattice_size, lattice_spacing)]),
         *np.arange(0, lattice_size, lattice_spacing)]
    for x in s:
        for y in s:
            for z in s:
                x_ = x - offset[0]
                y_ = y - offset[1]
                z_ = z - offset[2]
                yield x_, y_, z_
                X = x_ + lattice_spacing / 2
                Y = y_ + lattice_spacing / 2
                Z = z_ + lattice_spacing / 2
                if abs(X) <= lattice_size and abs(Y) <= lattice_size and abs(Z) <= lattice_size:
                    yield X, Y, Z


class Shape(ABC):
    """ Represents a shape """

    @abstractmethod
    def get_volume(self) -> float:
        pass

    @abstractmethod
    def get_region(self, name: str) -> str:
        pass

    @abstractmethod
    def get_lattice_point_count(self) -> int:
        pass


@dataclass
class Sphere(Shape):
    """ Represents a sphere """
    radius: float
    center: tuple

    def __init__(self, radius: float, center: tuple, check_in_box: bool = True):
        self.radius = radius
        self.center = center
        if check_in_box:
            self.assert_in_box()

    def assert_in_box(self):
        center = self.center
        radius = self.radius
        assert len(center) == 3, f"Center {self} must be a 3-tuple"
        assert (-BOX_SIZE <= center[0] <= BOX_SIZE and -BOX_SIZE <= center[1] <= BOX_SIZE and -BOX_SIZE <= center[
            2] <= BOX_SIZE), f"Sphere {self} too far from center"
        assert (-BOX_SIZE <= center[0] + radius <= BOX_SIZE and -BOX_SIZE <= center[
            1] + radius <= BOX_SIZE and -BOX_SIZE <= center[2] + radius <= BOX_SIZE), f"Sphere {self} is too big"
        assert (-BOX_SIZE <= center[0] - radius <= BOX_SIZE and -BOX_SIZE <= center[
            1] - radius <= BOX_SIZE and -BOX_SIZE <= center[2] - radius <= BOX_SIZE), f"Sphere {self} is too big"

    def get_volume(self) -> float:
        return 4.0 / 3.0 * pi * self.radius ** 3

    def get_region(self, name: str) -> str:
        return f"region {name} sphere {self.center[0]} {self.center[1]} {self.center[2]} {self.radius} units box"

    @staticmethod
    def from_volume(volume: float, center: tuple) -> 'Sphere':
        radius = (volume * 3 / (4 * pi)) ** (1 / 3)
        return Sphere(radius, center)

    @staticmethod
    def from_radius(radius: float, center: tuple) -> 'Sphere':
        return Sphere(radius, center)

    @staticmethod
    def from_lattice_point_count(lattice_point_count: int, center: tuple, tolerance: float = 1e-6) -> 'Sphere':
        def f(r) -> int:
            return Sphere(r, center).get_lattice_point_count() - lattice_point_count

        return Sphere(binary_search(f, 0, 100, tolerance), center)

    def get_lattice_point_count(self) -> int:
        count = 0
        for x, y, z in generate_bcc_lattice_points():
            if x ** 2 + y ** 2 + z ** 2 <= self.radius ** 2:
                count += 1
        return count


def __str__(self):
    return f"Sphere_{self.radius}_{self.center})"


@dataclass
class Plane(Shape):
    """ Represents a plane """
    point: tuple
    normal: tuple

    def __init__(self, point: tuple, normal: tuple):
        self.point = point
        self.normal = normal

    def get_volume(self) -> float:
        return 0.0

    def get_region(self, name: str) -> str:
        return f"region {name} plane {self.point[0]} {self.point[1]} {self.point[2]} {self.normal[0]} {self.normal[1]} {self.normal[2]} units box"

    def get_lattice_point_count(self) -> int:
        return 0

    def __str__(self):
        return f"Plane_{self.point}_{self.normal}"


@dataclass
class Cylinder(Shape):
    """ Represents a cylinder """
    radius: float
    length: float
    axis: str
    center: tuple

    def __init__(self, radius: float, length: float, axis: str, center: tuple, check_in_box: bool = True):
        self.radius = radius
        self.length = length
        self.axis = axis
        self.center = center
        if check_in_box:
            self.assert_inside_box()

    def get_volume(self) -> float:
        return pi * self.radius ** 2 * self.length

    def get_region(self, name: str) -> str:
        a = 0
        b = 0
        c = 0
        if self.axis == 'x':
            a = self.center[1]
            b = self.center[2]
            c = self.center[0]
        if self.axis == 'y':
            a = self.center[0]
            b = self.center[2]
            c = self.center[1]
        if self.axis == 'z':
            a = self.center[0]
            b = self.center[1]
            c = self.center[2]
        return f"region {name} cylinder {self.axis} {a} {b} {self.radius} {c - self.length / 2.0} {c + self.length / 2.0} units box"

    @staticmethod
    def from_volume_and_radius(volume: float, radius: float, axis: str, center: tuple) -> 'Cylinder':
        length = volume / (pi * radius ** 2)
        return Cylinder(radius, length, axis, center)

    @staticmethod
    def from_lattice_point_count_and_radius(lattice_point_count: int, radius: float, axis: str, center: tuple,
                                            tolerance: float = 1e-6) -> 'Cylinder':
        def f(leng) -> int:
            return Cylinder(radius, leng, axis, center,
                            check_in_box=False).get_lattice_point_count() - lattice_point_count

        return Cylinder(radius, binary_search(f, 0, 1000, tolerance), axis, center)

    @staticmethod
    def from_volume_and_length(volume: float, length: float, axis: str, center: tuple) -> 'Cylinder':
        radius = (volume / (pi * length)) ** 0.5
        return Cylinder(radius, length, axis, center)

    @staticmethod
    def from_radius_and_length(radius: float, length: float, axis: str, center: tuple) -> 'Cylinder':
        return Cylinder(radius, length, axis, center)

    def get_lattice_point_count(self) -> int:
        count = 0
        for x, y, z in generate_bcc_lattice_points():
            if self.axis == 'x':
                if y ** 2 + z ** 2 <= self.radius ** 2 and self.center[0] - self.length / 2 <= x <= self.center[
                    0] + self.length / 2:
                    count += 1
            if self.axis == 'y':
                if x ** 2 + z ** 2 <= self.radius ** 2 and self.center[1] - self.length / 2 <= y <= self.center[
                    1] + self.length / 2:
                    count += 1
            if self.axis == 'z':
                if x ** 2 + y ** 2 <= self.radius ** 2 and self.center[2] - self.length / 2 <= z <= self.center[
                    2] + self.length / 2:
                    count += 1
        return count

    def __str__(self):
        return f"Cylinder_{self.radius}_{self.length}_{self.axis}_{self.center}"

    def assert_inside_box(self):
        center = self.center
        radius = self.radius
        length = self.length
        assert len(center) == 3, f"Center {self} must be a 3-tuple"
        assert (-BOX_SIZE <= center[0] <= BOX_SIZE and -BOX_SIZE <= center[1] <= BOX_SIZE and -BOX_SIZE <= center[
            2] <= BOX_SIZE), f"Cylinder {self} too far form center"
        assert (-BOX_SIZE <= center[0] + radius <= BOX_SIZE and -BOX_SIZE <= center[
            1] + radius <= BOX_SIZE and -BOX_SIZE <= center[2] + radius <= BOX_SIZE), f"Cylinder {self} too much radius"
        assert (-BOX_SIZE <= center[0] - radius <= BOX_SIZE and -BOX_SIZE <= center[
            1] - radius <= BOX_SIZE and -BOX_SIZE <= center[2] - radius <= BOX_SIZE), f"Cylinder {self} too much radius"
        assert (-BOX_SIZE <= center[0] + length / 2 <= BOX_SIZE and -BOX_SIZE <= center[
            1] + length / 2 <= BOX_SIZE and -BOX_SIZE <= center[
                    2] + length / 2 <= BOX_SIZE), f"Cylinder {self} too long"
        assert (-BOX_SIZE <= center[0] - length / 2 <= BOX_SIZE and -BOX_SIZE <= center[
            1] - length / 2 <= BOX_SIZE and -BOX_SIZE <= center[
                    2] - length / 2 <= BOX_SIZE), f"Cylinder {self} too long"


@dataclass
class Cone(Shape):
    """ Represents a cone.
    cone args = dim c1 c2 radlo radhi lo hi
  dim = x or y or z = axis of cone
  c1,c2 = coords of cone axis in other 2 dimensions (distance units)
  radlo,radhi = cone radii at lo and hi end (distance units)
  lo,hi = bounds of cone in dim (distance units)
    """
    axis: str
    c1: float
    c2: float
    radlo: float
    radhi: float
    lo: float
    hi: float

    def __init__(self, axis: str, c1: float, c2: float, radlo: float, radhi: float, lo: float, hi: float):
        self.axis = axis
        self.c1 = c1
        self.c2 = c2
        self.radlo = radlo
        self.radhi = radhi
        self.lo = lo
        self.hi = hi

    def get_volume(self) -> float:
        return 0.0

    def get_region(self, name: str) -> str:
        return f"region {name} cone {self.axis} {self.c1} {self.c2} {self.radlo} {self.radhi} {self.lo} {self.hi} units box"

    def get_lattice_point_count(self) -> int:
        return 0

    def __str__(self):
        return f"Cone_{self.axis}_{self.c1}_{self.c2}_{self.radlo}_{self.radhi}_{self.lo}_{self.hi}"


@dataclass
class Prism(Shape):
    """Represents a prism
    prism args = xlo xhi ylo yhi zlo zhi xy xz yz
  xlo,xhi,ylo,yhi,zlo,zhi = bounds of untilted prism (distance units)
  xy = distance to tilt y in x direction (distance units)
  xz = distance to tilt z in x direction (distance units)
  yz = distance to tilt z in y direction (distance units)
    """
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float
    xy: float
    xz: float
    yz: float

    def __init__(self, xlo: float, xhi: float, ylo: float, yhi: float, zlo: float, zhi: float, xy: float, xz: float,
                 yz: float):
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi
        self.xy = xy
        self.xz = xz
        self.yz = yz

    def get_volume(self) -> float:
        return 0.0

    def get_region(self, name: str) -> str:
        return f"region {name} prism {self.xlo} {self.xhi} {self.ylo} {self.yhi} {self.zlo} {self.zhi} {self.xy} {self.xz} {self.yz} units box"

    def get_lattice_point_count(self) -> int:
        return 0

    def __str__(self):
        return f"Prism_{self.xlo}_{self.xhi}_{self.ylo}_{self.yhi}_{self.zlo}_{self.zhi}_{self.xy}_{self.xz}_{self.yz}"


@dataclass
class Ellipsoid(Shape):
    """Represents an ellipsoid
    ellipsoid args = x y z radiusx radiusy radiusz
    x,y,z = coords of ellipsoid center (distance units)
    radiusx,radiusy,radiusz = radii of ellipsoid (distance units)
    """
    x: float
    y: float
    z: float
    radiusx: float
    radiusy: float
    radiusz: float

    def __init__(self, x: float, y: float, z: float, radiusx: float, radiusy: float, radiusz: float):
        self.x = x
        self.y = y
        self.z = z
        self.radiusx = radiusx
        self.radiusy = radiusy
        self.radiusz = radiusz

    def get_volume(self) -> float:
        return 0.0

    def get_region(self, name: str) -> str:
        return f"region {name} ellipsoid {self.x} {self.y} {self.z} {self.radiusx} {self.radiusy} {self.radiusz} units box"

    def get_lattice_point_count(self) -> int:
        return 0

    def __str__(self):
        return f"Ellipsoid_{self.x}_{self.y}_{self.z}_{self.radiusx}_{self.radiusy}_{self.radiusz}"
