from unittest import TestCase

from nanoparticlebuilder import NanoparticleBuilder
import shapes
from poorly_coded_parser import PoorlyCodedParser
from shapes import Cylinder, Sphere


class TestPoorlyCodedParser(TestCase):
    def test_split_command(self):
        parts = PoorlyCodedParser.split_command('region\t\tHello world      123')
        self.assertEqual(['region', 'Hello', 'world', '123'], parts)
        parts = PoorlyCodedParser.split_command('delete_atoms from\tthere   ')
        self.assertEqual(['delete_atoms', 'from', 'there'], parts)

    def test_lattice(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("lattice bcc 2.8665", nano_builder)
        self.assertEqual(['lattice bcc 2.8665'], nano_builder.atom_manipulation)
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("lattice bcc 2.8665 origin 0 0 0.5", nano_builder)
        self.assertEqual(['lattice bcc 2.8665 origin 0 0 0.5'], nano_builder.atom_manipulation)

    def test_parse_group(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("group Ni type 2", nano_builder)
        self.assertEqual(['group Ni type 2'], nano_builder.atom_manipulation)
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("group \t\tFe type 1", nano_builder)
        self.assertEqual(['group Fe type 1'], nano_builder.atom_manipulation)

    def test_add_shape_0(self):
        nano_builder = self.confirm_shape(
            "region mycylinder cylinder x 0 0 10 -22.5 22.5 units box",
            "mycylinder",
            Cylinder(
                axis='x',
                radius=10,
                length=45,
                center=(0, 0, 0)
            )
        )
        PoorlyCodedParser.parse_line("group \t\tFe region mycylinder", nano_builder)
        self.assertEqual('group Fe region reg0', nano_builder.atom_manipulation[-1])

    def test_add_shape_1(self):
        self.confirm_shape(
            "region test sphere 0 0 0 10 units box",
            "test",
            Sphere(radius=10, center=(0, 0, 0))
        )

    def test_add_shape_2(self):
        self.confirm_shape(
            "region test plane 0 0 0 0 0 1 units box",
            "test",
            shapes.Plane((0, 0, 0), (0, 0, 1))
        )

    def test_add_shape_3(self):
        self.confirm_shape(
            "region test cone z 0 0 18 1 -21 21 units box",
            "test",
            shapes.Cone(
                'z',
                0,
                0,
                18,
                1,
                -21,
                21
            )
        )

    def test_add_shape_4(self):
        self.confirm_shape(
            "region test prism -20 20 -3 3 -16 16 0 0 0 units box",
            "test",
            shapes.Prism(-20, 20, -3, 3, -16, 16, 0, 0, 0)
        )

    def test_add_shape_5(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("region test sphere 0 0 0 10 units box", nano_builder)
        PoorlyCodedParser.parse_line("region test2 sphere 0 5 0 5 units box", nano_builder)
        PoorlyCodedParser.parse_line("region test intersect 2 test test2 units box", nano_builder)
        self.assertEqual("region reg2 intersect 2 reg0 reg1 units box", nano_builder.atom_manipulation[-1])

    def test_add_shape_6(self):
        self.confirm_shape(
            "region test ellipsoid 0 0 0 10 10 10 units box",
            "test",
            shapes.Ellipsoid(0, 0, 0, 10, 10, 10)
        )

    def test_parse_create_atoms(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("region test sphere 0 0 0 10 units box", nano_builder)
        PoorlyCodedParser.parse_line("create_atoms 1 region test", nano_builder)
        self.assertEqual("create_atoms 1 region reg0", nano_builder.atom_manipulation[-1])

    def test_parse_set(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("region test sphere 0 0 0 10 units box", nano_builder)
        PoorlyCodedParser.parse_line("set region test type 1", nano_builder)
        self.assertEqual(nano_builder.atom_manipulation[-1], "set region reg0 type 1")
        PoorlyCodedParser.parse_line("set region test type/subset 1 10 123", nano_builder)
        self.assertEqual(nano_builder.atom_manipulation[-1], "set region reg0 type/subset 1 10 zeed:0")
        PoorlyCodedParser.parse_line("set region test type/ratio 1 0.5 123", nano_builder)
        self.assertEqual(nano_builder.atom_manipulation[-1], "set region reg0 type/ratio 1 0.5 zeed:1")

        PoorlyCodedParser.parse_line("set group Fe type/subset 2 230 300", nano_builder)
        self.assertEqual(nano_builder.atom_manipulation[-1], "set group Fe type/subset 2 230 zeed:2")
        PoorlyCodedParser.parse_line("set group Fe type/ratio 2 0.2 300", nano_builder)
        self.assertEqual(nano_builder.atom_manipulation[-1], "set group Fe type/ratio 2 0.2 zeed:3")

    def test_parse_comment(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("# comment", nano_builder)
        self.assertEqual([], nano_builder.regions)
        self.assertEqual([], nano_builder.atom_manipulation)

    def test_delete_atoms(self):
        nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line("region test sphere 0 0 0 10 units box", nano_builder)
        PoorlyCodedParser.parse_line("delete_atoms region test compress yes", nano_builder)
        self.assertEqual('delete_atoms region reg0 compress yes', nano_builder.atom_manipulation[-1])

    def confirm_shape(
            self,
            command: str,
            name: str,
            shape: shapes.Shape,
            nano_builder: NanoparticleBuilder | None = None
    ) -> NanoparticleBuilder:
        if nano_builder is None:
            nano_builder: NanoparticleBuilder = NanoparticleBuilder()
        PoorlyCodedParser.parse_line(command, nano_builder)
        self.assertEqual([shape], nano_builder.regions)
        self.assertEqual(0, nano_builder.get_region_id_by_name(name))
        thrown = False
        try:
            self.assertEqual(-1, nano_builder.get_region_id_by_name("foo"))
        except KeyError:
            thrown = True
        self.assertTrue(thrown, "Should throw key error")
        return nano_builder
