import logging
import platform
import re
import subprocess

from template import get_slurm_template, replace_templates
from config import LAMMPS_EXECUTABLE, LAMMPS_TOKO_EXECUTABLE


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


def get_file_name(input_file):
	return "/".join(input_file.split("/")[-2:])


class MpiLammpsWrapper:
	@staticmethod
	def _simulate(
		input_file: str = "in.melt",
		gpu: GPUOpt = None,
		mpi: MPIOpt = None,
		omp: OMPOpt = None,
		in_toko: bool = False,
		cwd: str = './lammps_output',
	) -> bytes | None:
		if gpu is None: gpu = GPUOpt()
		if mpi is None: mpi = MPIOpt()
		if omp is None: omp = OMPOpt()
		if not in_toko:
			return MpiLammpsWrapper._simulate_in_local(gpu, mpi, omp, cwd, input_file)

		return MpiLammpsWrapper._simulate_in_toko(input_file)

	@staticmethod
	def _simulate_in_toko(input_file):
		MpiLammpsWrapper.confirm("Run in TOKO?")
		local_sim_folder = "/".join(input_file.split("/")[:-1])
		toko_sim_folder = "~/scratch/projects/magnetism/simulations/" + "/".join(get_file_name(input_file).split("/")[:-1])
		toko_nano_in = "~/scratch/projects/magnetism/simulations/" + get_file_name(input_file)
		logging.info("Creating simulation folder in toko...")
		MpiLammpsWrapper.mkdir_toko(toko_sim_folder)
		logging.info("Copying alloy files...")
		MpiLammpsWrapper.copy_file_to_toko("/".join(input_file.split("/")[:-3]) + "/FeCuNi.eam.alloy", "~/scratch/projects/magnetism/FeCuNi.eam.alloy")
		logging.info("Copying input file...")
		MpiLammpsWrapper.copy_file_to_toko(input_file, toko_nano_in)
		code = MpiLammpsWrapper.write_toko_script(toko_nano_in, toko_sim_folder, local_sim_folder)
		jobid = re.match(r"Submitted batch job (\d+)", code.decode('utf-8')).group(1)
		MpiLammpsWrapper.wait_for_execution(jobid)
		logging.info("Copying output files from toko to local machine...")
		MpiLammpsWrapper.copy_file_from_toko(toko_sim_folder, "/".join(local_sim_folder.split("/")[:-1]))
		logging.info(f"{code=}")

	@staticmethod
	def confirm(message):
		res = input(f"{message} (y/n) ")
		if res != "y":
			raise ValueError("Oki :c")

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

	@staticmethod
	def mkdir_toko(toko_path):
		return MpiLammpsWrapper.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"mkdir", f"{toko_path}"])

	@staticmethod
	def copy_file_to_toko(local_path, toko_path):
		return MpiLammpsWrapper.run_cmd_for_toko(lambda user, toko_url: ["scp", local_path, f"{user}@{toko_url}:{toko_path}"])

	@staticmethod
	def copy_file_from_toko(toko_path, local_path):
		return MpiLammpsWrapper.run_cmd_for_toko(lambda user, toko_url: ["scp", "-r", f"{user}@{toko_url}:{toko_path}", local_path])

	@staticmethod
	def run_cmd_for_toko(command_getter):
		user = "fwilliamson"
		toko_url = "toko.uncu.edu.ar"
		command = command_getter(user, toko_url)
		logging.debug(f"Running {command=}")
		return subprocess.check_output(command)

	@staticmethod
	def write_toko_script(toko_nano_in, toko_sim_folder, local_sim_folder):
		slurm_code = replace_templates(
			get_slurm_template(), {
				"lammps_exec": LAMMPS_TOKO_EXECUTABLE,
				"tasks": "1",
				"lammps_input": toko_nano_in,
				"lammps_output": toko_sim_folder + "/log.lammps",
				"cwd": toko_sim_folder,
				"partition": "mini"
			}
		)
		assert "{{" not in slurm_code, f"Not all templates were replaced in {slurm_code}"
		with open(local_sim_folder + "/slurm.sh", "w") as f:
			f.write(slurm_code)
		logging.info("Copying slurm.sh to toko...")
		MpiLammpsWrapper.copy_file_to_toko(local_sim_folder + "/slurm.sh", toko_sim_folder + "/slurm.sh")
		logging.info("Queueing job in toko...")
		return MpiLammpsWrapper.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'cd {toko_sim_folder}; /apps/slurm/bin/sbatch slurm.sh'"])

	@staticmethod
	def wait_for_execution(jobid):
		logging.info("Waiting for execution to finish in toko...")
		MpiLammpsWrapper.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'while [ \"$(/apps/slurm/bin/squeue -hj {jobid})\" != \"\" ]; do sleep 1; done'"])

	@staticmethod
	def _simulate_in_local(gpu, mpi, omp, cwd, input_file):
		lammps_executable = LAMMPS_EXECUTABLE
		cmd = f"{mpi} {lammps_executable} {omp} {gpu} -in {input_file}"
		cmd = re.sub(r' +', " ", cmd).strip()
		try:
			logging.info(f"Running [bold yellow]{cmd}[/bold yellow] in [cyan]{cwd}[/cyan]", extra={"markup": True, "highlighter": None})
			return subprocess.check_output(cmd.split(" "), cwd=cwd, shell=platform.system() == "Windows")
		except subprocess.CalledProcessError as e:
			MpiLammpsWrapper.print_error(cmd, cwd, e, gpu, False, input_file, lammps_executable, mpi, omp)
			raise e
		except OSError as e:
			MpiLammpsWrapper.print_error(cmd, cwd, e, gpu, False, input_file, lammps_executable, mpi, omp)
			raise ValueError(f"Is LAMMPS ({lammps_executable}) installed?") from e
