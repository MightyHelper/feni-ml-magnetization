import platform
import subprocess
import re
import logging
import time


class MPIOpt:
	use: bool = False
	hw_threads: bool = False
	n_threads: int = 1

	def __init__(self, use: bool = False, hw_threads: bool = False, n_threads: int = 1):
		self.use = use
		self.hw_threads = hw_threads
		self.n_threads = n_threads

	def __str__(self):
		return f"mpirun -n {self.n_threads} {'--use-hwthread-cpus' if self.hw_threads else ''} " if self.use else ""

	def __repr__(self):
		return f"MPIOpt(use={self.use}, hw_threads={self.hw_threads}, n_threads={self.n_threads})"


class GPUOpt:
	use: bool = False

	def __init__(self, use: bool = False):
		self.use = use

	def __str__(self):
		return "-sf gpu -pk gpu 1" if self.use else ""

	def __repr__(self):
		return f"GPUOpt(use={self.use})"


class OMPOpt:
	use: bool = False
	n_threads: int = 1

	def __init__(self, use: bool = False, n_threads: int = 1):
		self.use = use
		self.n_threads = n_threads

	def __str__(self):
		return f"-sf omp -pk omp {self.n_threads}" if self.use else ""

	def __repr__(self):
		return f"OMPOpt(use={self.use}, n_threads={self.n_threads})"


class MpiLammpsWrapper:
	@staticmethod
	def _simulate(
		input_file: str = "in.melt",
		gpu: GPUOpt = None,
		mpi: MPIOpt = None,
		omp: OMPOpt = None,
		in_toko: bool = False,
		cwd: str = './lammps_output',
		lammps_executable: str = 'lmp'
	) -> bytes | None:
		if gpu is None: gpu = GPUOpt()
		if mpi is None: mpi = MPIOpt()
		if omp is None: omp = OMPOpt()
		cmd = f"{mpi} {lammps_executable} {omp} {gpu} -in {input_file}"
		cmd = re.sub(r' +', " ", cmd).strip()
		if not in_toko:
			try:
				logging.info(f"Running [bold yellow]{cmd}[/bold yellow] in [cyan]{cwd}[/cyan]", extra={"markup": True, "highlighter": None})
				return subprocess.check_output(cmd.split(" "), cwd=cwd, shell=platform.system() == "Windows")
			except subprocess.CalledProcessError as e:
				MpiLammpsWrapper.print_error(cmd, cwd, e, gpu, in_toko, input_file, lammps_executable, mpi, omp)
				raise e
			except OSError as e:
				MpiLammpsWrapper.print_error(cmd, cwd, e, gpu, in_toko, input_file, lammps_executable, mpi, omp)
				raise ValueError(f"Is LAMMPS ({lammps_executable}) installed?") from e
		return print("TODO: TOKO not implemented yet :c")

	@staticmethod
	def print_error(cmd, cwd, e, gpu, in_toko, input_file, lammps_executable, mpi, omp):
		logging.error(
			"ERROR:" +
			str(e) +
			f"\nERROR: {input_file=}" +
			f"\nERROR: {repr(gpu)=} " +
			f"\nERROR: {repr(mpi)=} " +
			f"\nERROR: {repr(omp)=} " +
			f"\nERROR: {in_toko=}" +
			f"\nERROR: {cwd=}" +
			f"\nERROR: {''.join(cmd)=}" +
			f"\nERROR: {lammps_executable=}",
			extra={"markup": True}
		)

	@staticmethod
	def gen_and_sim(code: str, sim_params: dict = None, file_to_use: str = '/tmp/in.melt') -> str:
		assert "{{" not in code, "Not all templates were replaced"
		with open(file_to_use, 'w') as f:
			f.write(code)
		result = MpiLammpsWrapper._simulate(input_file=file_to_use, **sim_params)
		if result is None:
			return ""
		return result.decode('utf-8')
