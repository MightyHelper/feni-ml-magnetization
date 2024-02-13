import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

DistributionType = StrEnum("DistributionType", ['RANDOM', 'SANDWICH', 'MULTILAYER', 'ONION'])
ShapeType = StrEnum("ShapeType", ['SPHERE', 'CUBE', 'CYLINDER', 'CONE', 'CROSS', 'ELLIPSOID', 'FLAKE', 'OCTAHEDRON', 'PYRAMID', 'TOROID'])


class ShapeNameError(ValueError):
    pass


@dataclass
class Distribution:
    type: DistributionType
    count: int
    params: list[str]

    @staticmethod
    def of(param: str) -> 'Distribution':
        parts = param.split(".")
        typ: str = parts[0]
        if len(parts) == 1:
            return Distribution(DistributionType[typ.upper()], 0, [])
        try:
            count: int = int(parts[1])
        except ValueError:
            match = re.match(r"(\d+)\[(.*)]", parts[1])
            assert match is not None, f"Invalid distribution: {param}, {parts[1]}"
            count = int(match.group(1))
            params = [match.group(2), *parts[2:]]
            return Distribution(DistributionType[typ.upper()], count, params)
        params: list[str] = parts[2:]
        return Distribution(DistributionType[typ.upper()], count, params)


@dataclass
class Shape:
    type: ShapeType

    @staticmethod
    def of(param: str) -> 'Shape':
        try:
            upper: str = param.upper()
            return Shape(ShapeType[upper])
        except KeyError as e:
            raise ShapeNameError(f"Invalid shape: {param}") from e


@dataclass
class Interface:
    mix_level: float

    @staticmethod
    def of(param: str) -> 'Interface':
        if param == "Normal":
            return Interface(0.0)
        elif param.startswith("Mix"):
            try:
                return Interface(float(param[4:]))
            except ValueError as e:
                raise ShapeNameError(f"Invalid interface: {param}") from e
        else:
            raise ShapeNameError(f"Invalid interface: {param}")


@dataclass
class Pores:
    count: int
    size: float | None

    @staticmethod
    def of(param: str) -> 'Pores':
        if param == "Full":
            return Pores(0, None)
        elif param.startswith("Pores."):
            match = re.match(r"(\d+)(\[(.*)])?", param.split(".")[1])
            assert match is not None, f"Invalid pores: {param}"
            count = int(match.group(1))
            size = float(match.group(3)) if match.group(3) else None
            return Pores(count, size)
        else:
            raise ShapeNameError(f"Invalid pores: {param}")


@dataclass
class NanoparticleData:
    shape: Shape
    distribution: Distribution
    interface: Interface
    pores: Pores
    index: int
    random_seed: int | None


def get_line_index_startswith(lines: list[str], prefix: str) -> int:
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index
    raise AssertionError(f"Line starting with {prefix} not found")


@dataclass
class NanoparticleFile:
    name: str
    path: Path

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem

    @cached_property
    def data(self) -> NanoparticleData:
        try:
            parts: list[str] = self.name.split("_")
            assert len(parts) in (5, 6)
            shape: Shape = Shape.of(parts[0])
            distribution: Distribution = Distribution.of(parts[1])
            interface: Interface = Interface.of(parts[2])
            pores: Pores = Pores.of(parts[3])
            index = int(parts[4])
            random_seed = int(parts[5]) if len(parts) == 6 else None
            return NanoparticleData(shape, distribution, interface, pores, index, random_seed)
        except (ShapeNameError, AssertionError) as e:
            raise ShapeNameError(f"Invalid nanoparticle file name: {self.name}") from e

    @cached_property
    def content(self):
        return self.path.read_text()

    @cached_property
    def lines(self):
        return self.content.split("\n")

    @cached_property
    def snippet(self):
        start = get_line_index_startswith(self.lines, "lattice")
        try:
            end = get_line_index_startswith(self.lines, "# setting")
        except AssertionError:
            end = get_line_index_startswith(self.lines, "mass")
        return self.lines[start:end]
