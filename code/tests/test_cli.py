import os.path
import shutil
import unittest
from pathlib import Path

import cli_parts.executions as executions
import cli_parts.shapefolder as shapefolder
import nanoparticle_locator
from nanoparticle import Nanoparticle


class TestShapefolder(unittest.TestCase):
    def test_ls(self):
        shapefolder.ls()

    # def test_shrink(self):
    #     shapefolder.shrink()

    def test_parse_all(self):
        expected: int = len([*nanoparticle_locator.NanoparticleLocator.search("../Shapes", ".in")])
        actual: list[tuple[str, Nanoparticle]] = shapefolder.parseshapes(test=True, seed_count=1, at="local:16")
        ok = [x for x, y in actual if y.is_ok()]
        self.assertEqual(expected, len(actual), "Lost some executions")
        self.assertEqual(expected, len(ok), "Some executions failed")


class TestExec(unittest.TestCase):
    def test_ls(self):
        executions.ls()

    def test_execute(self):
        ironsphere_in = "../Shapes/Test/Cone_Multilayer.2.Axis.X_05_Full_0.in"
        execution_result: str | None = executions.execute(
            path=Path(ironsphere_in),
            plot=False,
            test=True,
            at="local"
        )
        if execution_result is None:
            self.fail("Execution failed :c")
        self.assertTrue(os.path.exists(execution_result), "Execution doesnt exist")
        self.assertTrue(os.path.exists(os.path.join(execution_result, "log.lammps")), "No Lammps log")
        self.assertTrue(os.path.exists(os.path.join(execution_result, "iron.0.dump")), "No dump")
        nano = Nanoparticle.from_executed(execution_result)
        expected = {
            'ok': True,
            'fe': 830,
            'ni': 425,
            'total': 1255,
            'ratio_fe': 0.6613545816733067,
            'ratio_ni': 0.3386454183266932
        }
        result = nano.asdict()
        for key, value in expected.items():
            self.assertEqual(value, result[key])
        self.assertIn(ironsphere_in, nano.title)
        shutil.rmtree(execution_result)
