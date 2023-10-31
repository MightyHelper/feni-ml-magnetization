import shapes
import mpilammpsrun as mpilr
import mpilammpswrapper as mpilw
import template


class Nanoparticle:
	""" Represents a nanoparticle """
	shape: shapes.Shape
	run: mpilr.MpiLammpsRun

	def __init__(self, shape: shapes.Shape, extra_replacements: dict = None):
		self.shape = shape
		self.extra_replacements = {} if extra_replacements is None else extra_replacements

	def execute(self, test_run: bool = True, **kwargs):
		code = template.replace_templates(template.get_template(), {
			"region": self.shape.get_region(),
			"run_steps": str(0 if test_run else 300000),
			**self.extra_replacements
		})
		self.run = mpilr.MpiLammpsRun(
			code,
			{
				"lammps_executable": template.LAMMPS_EXECUTABLE,
				"omp": mpilw.OMPOpt(use=True, n_threads=2),
				"mpi": mpilw.MPIOpt(use=True, hw_threads=False, n_threads=4),
				**kwargs
			},
			[f"iron.{i}.dump" for i in ([0] if test_run else [0, 100000, 200000, 300000])]
		).execute()
