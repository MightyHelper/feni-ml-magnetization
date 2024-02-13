import hashlib
from pathlib import Path
from typing import TypeVar

import nanoparticlefile as f

T = TypeVar("T")


def assert_equal(a: T, b: T, message: str | None = None):
    assert a == b, message if message is not None else f"Expected {a}, got {b}"


def test_distribution():
    tests: dict[str, tuple[f.DistributionType, int, list[str]]] = {
        "Onion.2": (f.DistributionType.ONION, 2, []),
        "Onion.2[Cone]": (f.DistributionType.ONION, 2, ["Cone"]),
        "Onion.3": (f.DistributionType.ONION, 3, []),
        "Sandwich.2.X": (f.DistributionType.SANDWICH, 2, ["X"]),
        "Random": (f.DistributionType.RANDOM, 0, []),
    }
    for name, (typ, count, params) in tests.items():
        distribution: f.Distribution = f.Distribution.of(name)
        assert_equal(distribution.type, typ)
        assert_equal(distribution.count, count)
        assert_equal(distribution.params, params)


def test_shape():
    tests: dict[str, f.ShapeType] = {
        "Cone": f.ShapeType.CONE,
        "CUBE": f.ShapeType.CUBE,
        "Sphere": f.ShapeType.SPHERE,
    }
    for name, shape in tests.items():
        assert_equal(f.Shape.of(name).type, shape, f"Expected {shape}, got {f.Shape.of(name).type}")


def test_interface():
    tests: dict[str, float] = {
        "Mix.05": 5,
        "Normal": 0,
        "Mix.10": 10,
    }
    for name, mix_level in tests.items():
        assert_equal(f.Interface.of(name).mix_level, mix_level)


def test_nanoparticle_file_name():
    tests: dict[str, tuple[f.DistributionType, int, list[str], f.ShapeType, float, int, int]] = {
        "../Shapes/Cone_Random_Normal_Full_0.in": (f.DistributionType.RANDOM, 0, [], f.ShapeType.CONE, 0.0, 0, None),
        "../Shapes/Cone_Random_Normal_Pores.1_0_123.in": (f.DistributionType.RANDOM, 0, [], f.ShapeType.CONE, 0.0, 0, 123),
        "../Shapes/Sphere_Onion.2[Cone]_Mix.05_Pores.2[7.2A]_1.in": (f.DistributionType.ONION, 2, ["Cone"], f.ShapeType.SPHERE, 5.0, 1, None),
        "../Shapes/Cube_Sandwich.2.X_Mix.10_Full_2.in": (f.DistributionType.SANDWICH, 2, ["X"], f.ShapeType.CUBE, 10.0, 2, None),
        "../Shapes/Octahedron_Random_Normal_Full_0_3.in": (f.DistributionType.RANDOM, 0, [], f.ShapeType.OCTAHEDRON, 0.0, 0, 3),
    }
    for name, (distribution_type, count, params, shape_type, mix_level, index, random_seed) in tests.items():
        nanoparticle_file: f.NanoparticleFile = f.NanoparticleFile(Path(name))
        nanoparticle_data: f.NanoparticleData = nanoparticle_file.data
        assert_equal(nanoparticle_data.shape.type, shape_type)
        assert_equal(nanoparticle_data.distribution.type, distribution_type)
        assert_equal(nanoparticle_data.distribution.count, count)
        assert_equal(nanoparticle_data.distribution.params, params)
        assert_equal(nanoparticle_data.interface.mix_level, mix_level)
        assert_equal(nanoparticle_data.index, index)
        assert_equal(nanoparticle_data.random_seed, random_seed)


def test_nanoparticle_snippet():
    files: dict[str, str] = {
        "../Shapes/Cone_Random_Normal_Full_0.in": "77a20abbfa50625df0f2257a2d9644a1",
        "../Shapes/Cone_Random_Normal_Pores.2_0.in": "7e25e6bd57fedaedaf12b70678c7ce1c",
    }
    for name, index in files.items():
        nanoparticle_file: f.NanoparticleFile = f.NanoparticleFile(Path(name))
        snippet: str = "\n".join(nanoparticle_file.snippet)
        snippet_hash: str = hashlib.md5(snippet.encode()).hexdigest()
        assert_equal(snippet_hash, index, f"Expected {index}, got {snippet_hash}")


if __name__ == "__main__":
    test_distribution()
    test_shape()
    test_interface()
    test_nanoparticle_file_name()
    test_nanoparticle_snippet()
    print("nanoparticlefile_test.py: all tests pass")
