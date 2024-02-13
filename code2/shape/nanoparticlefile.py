import re
from dataclasses import dataclass
from enum import StrEnum
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
    def parse(param: str) -> 'Distribution':
        parts = param.split(".")
        typ: str = parts[0]
        if len(parts) == 1:
            return Distribution(DistributionType[typ.upper()], 0, [])
        try:
            count: int = int(parts[1])
        except ValueError:
            match = re.match(r"(\d+)\[(.*)\]", parts[1])
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
    def parse(param: str) -> 'Shape':
        try:
            upper: str = param.upper()
            return Shape(ShapeType[upper])
        except KeyError as e:
            raise ShapeNameError(f"Invalid shape: {param}") from e


@dataclass
class Interface:
    mix_level: float

    @staticmethod
    def parse(param: str) -> 'Interface':
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
class NanoparticleData:
    shape: Shape
    distribution: Distribution
    interface: Interface
    index: int
    random_seed: int | None


@dataclass
class NanoparticleFile:
    name: str
    path: Path

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem

    def parse(self) -> NanoparticleData:
        try:
            parts: list[str] = self.name.split("_")
            assert len(parts) in (4, 5)
            shape: Shape = Shape.parse(parts[0])
            distribution: Distribution = Distribution.parse(parts[1])
            interface: Interface = Interface.parse(parts[2])
            index = int(parts[3])
            random_seed = int(parts[4]) if len(parts) == 5 else None
            return NanoparticleData(shape, distribution, interface, index, random_seed)
        except (ShapeNameError, AssertionError) as e:
            raise ShapeNameError(f"Invalid nanoparticle file name: {self.name}") from e
