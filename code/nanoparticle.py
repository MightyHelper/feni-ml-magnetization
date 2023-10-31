import subprocess
import time
import os
import shapes
import mpilammpsrun as mpilr
import mpilammpswrapper as mpilw
import template
import feni_mag


def realpath(path):
	return subprocess.check_output(["realpath", path]).decode("ascii").strip()


class Nanoparticle:
	"""Represents a nanoparticle"""
	shape: shapes.Shape
	run: mpilr.MpiLammpsRun

	def __init__(self, shape: shapes.Shape, extra_replacements: dict = None):
		self.shape = shape
		self.extra_replacements = {} if extra_replacements is None else extra_replacements
		self.path = realpath("../executions/" + self.get_identifier()) + "/"
		print(f"Path: {self.path}")

	def execute(self, test_run: bool = True, **kwargs):
		code = template.replace_templates(template.get_template(), {
			"region": self.shape.get_region(),
			"run_steps": str(0 if test_run else 300000),
			**self.extra_replacements
		})
		os.mkdir(self.path)
		self.run = mpilr.MpiLammpsRun(
			code,
			{
				"lammps_executable": template.LAMMPS_EXECUTABLE,
				"omp": mpilw.OMPOpt(use=True, n_threads=2),
				"mpi": mpilw.MPIOpt(use=True, hw_threads=False, n_threads=4),
				"cwd": self.path,
				**kwargs
			},
			[f"iron.{i}.dump" for i in ([0] if test_run else [0, 100000, 200000, 300000])],
			file_name=self.path + "nanoparticle.in"
		).execute()
		feni_mag.extract_magnetism(self.path + "/log.lammps", out_mag=self.path + "/magnetism.txt", digits=4)

	def get_identifier(self):
		return f"simulation_{time.time()}"
