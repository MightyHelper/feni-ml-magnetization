import os
from pathlib import Path
from typing import Any
from opt import MPIOpt, GPUOpt, OMPOpt
from simulation_task import SimulationTask


class SimulationWrapper:
	"""
	Wrapper for MPI LAMMPS
	"""

	@staticmethod
	def get_task(
		input_file: str = "in.melt",
		gpu: GPUOpt = None,
		mpi: MPIOpt = None,
		omp: OMPOpt = None,
		cwd: str = './lammps_output',
	) -> SimulationTask:
		if gpu is None: gpu = GPUOpt()
		if mpi is None: mpi = MPIOpt()
		if omp is None: omp = OMPOpt()
		return SimulationTask(input_file, gpu, mpi, omp, cwd)

	@staticmethod
	def generate(code: str, sim_params: dict[str, Any] = None, file_to_use: str = '/tmp/in.melt') -> SimulationTask:
		"""
		Generate the local folder structure and return a simulation task
		:param code:
		:param sim_params:
		:param file_to_use:
		:return:
		"""
		assert "{{" not in code, "Not all templates were replaced"
		path = Path(file_to_use)
		if not path.parent.exists():
			if not path.parent.parent.exists():
				os.mkdir(path.parent.parent)
			os.mkdir(path.parent)
		with open(file_to_use, 'w') as f: f.write(code)
		return SimulationWrapper.get_task(input_file=file_to_use, **sim_params)
