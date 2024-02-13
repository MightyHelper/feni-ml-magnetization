from pathlib import Path

import nanoparticlefile as f


def test_distribution():
    tests: dict[str, tuple[f.DistributionType, int, list[str]]] = {
        "Onion.2": (f.DistributionType.ONION, 2, []),
        "Onion.2[Cone]": (f.DistributionType.ONION, 2, ["Cone"]),
        "Onion.3": (f.DistributionType.ONION, 3, []),
        "Sandwich.2.X": (f.DistributionType.SANDWICH, 2, ["X"]),
        "Random": (f.DistributionType.RANDOM, 0, []),
    }
    for name, (typ, count, params) in tests.items():
        distribution = f.Distribution.parse(name)
        assert distribution.type == typ
        assert distribution.count == count
        assert distribution.params == params


def test_shape():
    tests: dict[str, f.ShapeType] = {
        "Cone": f.ShapeType.CONE,
        "CUBE": f.ShapeType.CUBE,
        "Sphere": f.ShapeType.SPHERE,
    }
    for name, shape in tests.items():
        assert f.Shape.parse(name).type == shape, f"Expected {shape}, got {f.Shape.parse(name).type}"


def test_interface():
    tests: dict[str, float] = {
        "Mix.05": 5,
        "Normal": 0,
        "Mix.10": 10,
    }
    for name, mix_level in tests.items():
        assert f.Interface.parse(name).mix_level == mix_level


def test_nanoparticlefile():
    tests: dict[str, tuple[f.DistributionType, int, list[str], f.ShapeType, float, int, int]] = {
        "../Shapes/Cone_Random_Normal_0.in": (f.DistributionType.RANDOM, 0, [], f.ShapeType.CONE, 0.0, 0, None),
        "../Shapes/Cone_Random_Normal_0_123.in": (f.DistributionType.RANDOM, 0, [], f.ShapeType.CONE, 0.0, 0, 123),
        "../Shapes/Sphere_Onion.2[Cone]_Mix.05_1.in": (f.DistributionType.ONION, 2, ["Cone"], f.ShapeType.SPHERE, 5.0, 1, None),
        "../Shapes/Cube_Sandwich.2.X_Mix.10_2.in": (f.DistributionType.SANDWICH, 2, ["X"], f.ShapeType.CUBE, 10.0, 2, None),
        "../Shapes/Octahedron_Random_Normal_0_3.in": (f.DistributionType.RANDOM, 0, [], f.ShapeType.OCTAHEDRON, 0.0, 0, 3),
    }
    for name, (distribution_type, count, params, shape_type, mix_level, index, random_seed) in tests.items():
        nanoparticle_file = f.NanoparticleFile(Path(name))
        nanoparticle_data = nanoparticle_file.parse()
        assert nanoparticle_data.shape.type == shape_type, f"Expected {shape_type}, got {nanoparticle_data.shape.type}"
        assert nanoparticle_data.distribution.type == distribution_type, f"Expected {distribution_type}, got {nanoparticle_data.distribution.type}"
        assert nanoparticle_data.distribution.count == count, f"Expected {count}, got {nanoparticle_data.distribution.count}"
        assert nanoparticle_data.distribution.params == params, f"Expected {params}, got {nanoparticle_data.distribution.params}"
        assert nanoparticle_data.interface.mix_level == mix_level, f"Expected {mix_level}, got {nanoparticle_data.interface.mix_level}"
        assert nanoparticle_data.index == index, f"Expected {index}, got {nanoparticle_data.index}"
        assert nanoparticle_data.random_seed == random_seed, f"Expected {random_seed}, got {nanoparticle_data.random_seed}"


if __name__ == "__main__":
    test_distribution()
    test_shape()
    test_interface()
    test_nanoparticlefile()
    print("nanoparticlefile_test.py: all tests pass")
