import os
from typing import Any
from opt import MPIOpt, GPUOpt, OMPOpt
from simulation_task import SimulationTask


class MpiLammpsWrapper:
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
		assert "{{" not in code, "Not all templates were replaced"
		assert os.path.exists(os.path.dirname(file_to_use)), f"Folder {os.path.dirname(file_to_use)} does not exist"
		with open(file_to_use, 'w') as f: f.write(code)
		return MpiLammpsWrapper.get_task(input_file=file_to_use, **sim_params)
