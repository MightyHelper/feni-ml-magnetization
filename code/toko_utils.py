import logging
import os
import random
import re
import subprocess
import time
from pathlib import Path

import config
from config import LAMMPS_TOKO_EXECUTABLE, TOKO_PARTITION_TO_USE, TOKO_USER, TOKO_URL, TOKO_EXECUTION_PATH, LAMMPS_EXECUTABLE
from execution_queue import ExecutionQueue
from simulation_task import SimulationTask
from template import TemplateUtils
from utils import confirm, write_local_file


class TokoUtils:
	@staticmethod
	def mkdir_toko(toko_path):
		return TokoUtils.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"mkdir", f"{toko_path}"])

	@staticmethod
	def copy_file_to_toko(local_path: str, toko_path: str, is_folder: bool = False):
		return TokoUtils.run_cmd_for_toko(
			lambda user, toko_url:
			["rsync", "-r", local_path + "/", f"{user}@{toko_url}:{toko_path}/"] if is_folder else
			["rsync", local_path, f"{user}@{toko_url}:{toko_path}"]
		)

	@staticmethod
	def copy_file_from_toko(toko_path: str, local_path: str, is_folder: bool = False):
		return TokoUtils.run_cmd_for_toko(
			lambda user, toko_url:
			["rsync", "-r", f"{user}@{toko_url}:{toko_path}/", local_path + "/"] if is_folder else
			["rsync", f"{user}@{toko_url}:{toko_path}", local_path]
		)

	@staticmethod
	def run_cmd_for_toko(command_getter):
		command = command_getter(TOKO_USER, TOKO_URL)
		logging.debug(f"Running {command=}")
		return subprocess.check_output(command)

	@staticmethod
	def wait_for_toko_execution(jobid):
		logging.info("Waiting for execution to finish in toko...")
		TokoUtils.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'while [ \"$(/apps/slurm/bin/squeue -hj {jobid})\" != \"\" ]; do sleep 1; done'"])

	@staticmethod
	def copy_alloy_files(local_sim_folder, toko_sim_folder):
		local_alloy_file: Path = Path(os.path.join(local_sim_folder.parent.parent, "FeCuNi.eam.alloy"))
		toko_alloy_file: Path = Path(os.path.join(toko_sim_folder.parent.parent, "FeCuNi.eam.alloy"))
		logging.info("Copying alloy files...")
		TokoUtils.copy_file_to_toko(local_alloy_file.as_posix(), toko_alloy_file.as_posix())


class TokoExecutionQueue(ExecutionQueue):
	@staticmethod
	def submit_toko_script(toko_nano_in: str, toko_sim_folder: str, local_sim_folder: str):
		slurm_code = TemplateUtils.replace_templates(
			TemplateUtils.get_slurm_template(), {
				"lammps_exec": LAMMPS_TOKO_EXECUTABLE,
				"tasks": "1",
				"time": "00:45:00",
				"lammps_input": toko_nano_in,
				"lammps_output": toko_sim_folder + "/log.lammps",
				"cwd": toko_sim_folder,
				"partition": TOKO_PARTITION_TO_USE
			}
		)
		assert "{{" not in slurm_code, f"Not all templates were replaced in {slurm_code}"
		local_slurm_sh = os.path.join(local_sim_folder, config.SLURM_SH)
		write_local_file(local_slurm_sh, slurm_code)
		logging.info(f"Copying {config.SLURM_SH} to toko...")
		toko_slurm_sh = os.path.join(toko_sim_folder, config.SLURM_SH)
		TokoUtils.copy_file_to_toko(local_slurm_sh, toko_slurm_sh)
		logging.info("Queueing job in toko...")
		return TokoUtils.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'cd {toko_sim_folder}; {config.TOKO_SBATCH} {config.SLURM_SH}'"])

	@staticmethod
	def simulate_in_toko(simulation_task: SimulationTask) -> str:
		input_path = Path(simulation_task.input_file)
		local_sim_folder: Path = input_path.parent.absolute()
		toko_sim_folder: Path = Path(os.path.join(TOKO_EXECUTION_PATH, input_path.parent.name))
		toko_nano_in: Path = Path(os.path.join(toko_sim_folder, input_path.name))
		logging.info("Creating simulation folder in toko...")
		logging.debug(f"{toko_sim_folder=}")
		logging.debug(f"{local_sim_folder=}")
		logging.debug(f"{toko_nano_in=}")
		logging.debug("Copying input file...")
		TokoUtils.copy_file_to_toko(local_sim_folder.as_posix(), toko_sim_folder.as_posix(), is_folder=True)
		TokoUtils.copy_alloy_files(local_sim_folder, toko_sim_folder)
		sbatch_output = TokoExecutionQueue.submit_toko_script(toko_nano_in.as_posix(), toko_sim_folder.as_posix(), local_sim_folder.as_posix())
		jobid = re.match(r"Submitted batch job (\d+)", sbatch_output.decode('utf-8')).group(1)
		TokoUtils.wait_for_toko_execution(jobid)
		logging.info("Copying output files from toko to local machine...")
		TokoUtils.copy_file_from_toko(toko_sim_folder.as_posix(), local_sim_folder.as_posix(), is_folder=True)
		local_lammps_log = os.path.join(local_sim_folder, "log.lammps")
		with open(local_lammps_log, "r") as f:
			lammps_log = f.read()
		return lammps_log

	def _simulate(self) -> None:
		simulation_task: SimulationTask = self._get_next_task()
		try:
			logging.info(f"[bold green]TokoExecutionQueue[/bold green] Running [bold yellow]{simulation_task.input_file}[/bold yellow] in [cyan]{simulation_task.cwd}[/cyan]", extra={"markup": True, "highlighter": None})
			result = TokoExecutionQueue.simulate_in_toko(simulation_task)
			self.run_callback(simulation_task, result)
		except subprocess.CalledProcessError as e:
			self.print_error(e)
			raise e
		except OSError as e:
			self.print_error(e)
			raise ValueError(f"Is LAMMPS ({LAMMPS_EXECUTABLE}) installed?") from e


def estimate_time(count, tasks):
	"""
	ceil(Count / tasks) * 45 min in hh:mm:ss
	Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
	:param count:
	:param tasks:
	:return:
	"""
	minutes = int((count / tasks) * 45)
	hours = minutes // 60
	days = hours // 24
	remaining_minutes = minutes % 60
	remaining_hours = hours % 24
	return f"{days}-{remaining_hours}:{remaining_minutes}"


class TokoBatchedExecutionQueue(ExecutionQueue):
	def __init__(self, batch_size: int = 10):
		super().__init__()
		self.batch_size = batch_size

	@staticmethod
	def submit_toko_script(simulation_info: list[tuple[str, str, str]], local_batch_path: str, toko_batch_path: str, n_tasks: int = 1):
		tasks = []
		# Create a list of commands to run in parallel
		for toko_nano_in, toko_sim_folder, local_sim_folder in simulation_info:
			lammps_log = os.path.join(toko_sim_folder, "log.lammps")
			tasks.append(f"\"sh -c 'cd {toko_sim_folder}; {LAMMPS_TOKO_EXECUTABLE} -in {toko_nano_in} > {lammps_log}'\"")
		# Create a slurm script to run the commands in parallel
		slurm_code = TemplateUtils.replace_templates(
			TemplateUtils.get_slurm_multi_template(), {
				"tasks": str(n_tasks),
				"time": estimate_time(len(tasks), n_tasks),
				"cmds": " ".join(tasks),
				"cwd": toko_batch_path,
				"partition": TOKO_PARTITION_TO_USE,
				"output": os.path.join(toko_batch_path, "batch_run.out"),
			}
		)
		assert "{{" not in slurm_code, f"Not all templates were replaced in {slurm_code}"
		local_slurm_sh = os.path.join(local_batch_path, config.SLURM_SH)
		write_local_file(local_slurm_sh, slurm_code)
		logging.info(f"Copying {config.SLURM_SH} to toko...")
		toko_slurm_sh = os.path.join(toko_batch_path, config.SLURM_SH)
		TokoUtils.copy_file_to_toko(local_slurm_sh, toko_slurm_sh)
		logging.info("Queueing job in toko...")
		return TokoUtils.run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'cd {toko_batch_path}; {config.TOKO_SBATCH} {config.SLURM_SH}'"])

	def _simulate(self):
		simulations = self._get_next_task()
		local_batch_path, toko_batch_path, simulation_info = self.prepare_scripts(simulations)
		sbatch_output = TokoBatchedExecutionQueue.submit_toko_script(simulation_info, local_batch_path, toko_batch_path)
		self.process_output(local_batch_path, sbatch_output, simulations, toko_batch_path)

	def prepare_scripts(self, simulations):
		"""
		Create a folder in toko with the simulations and a multi.py file to run them in parallel
		Then, copy the folder to toko
		:param simulations: List of simulations to run
		:return:
		"""
		simulation_info = []
		# Make new dir in TOKO_EXECUTION_PATH "batch_<timestamp>_<random[0-1000]>"
		batch_name = f"batch_{int(time.time())}_{random.randint(0, 1000)}"
		local_batch_path = os.path.join(config.LOCAL_EXECUTION_PATH, batch_name)
		toko_batch_path = os.path.join(TOKO_EXECUTION_PATH, batch_name)
		batch_info = os.path.join(local_batch_path, "batch_info.txt")
		local_multi_py = config.LOCAL_MULTI_PY
		toko_multi_py = os.path.join(toko_batch_path, "multi.py")
		os.mkdir(local_batch_path)
		write_local_file(batch_info, "\n".join([f"{i + 1}: {simulation.input_file}" for i, simulation in enumerate(simulations)]))
		for i, simulation in enumerate(simulations):
			local_sim_folder: Path = Path(simulation.input_file).parent.absolute()
			toko_sim_folder: Path = Path(os.path.join(TOKO_EXECUTION_PATH, local_sim_folder.name))
			toko_nano_in: Path = Path(os.path.join(toko_sim_folder, Path(simulation.input_file).name))
			logging.info(f"Creating simulation folder in toko ({i + 1}/{len(simulations)})...")
			logging.debug(f"{toko_sim_folder=}")
			logging.debug(f"{local_sim_folder=}")
			logging.debug(f"{toko_nano_in=}")
			logging.debug("Copying input file...")
			TokoUtils.copy_file_to_toko(local_sim_folder.as_posix(), toko_sim_folder.as_posix(), is_folder=True)
			if i == 0:
				TokoUtils.copy_alloy_files(local_sim_folder, toko_sim_folder)
			simulation_info.append((toko_nano_in.as_posix(), toko_sim_folder.as_posix(), local_sim_folder.as_posix()))
		TokoUtils.copy_file_to_toko(local_batch_path, toko_batch_path, is_folder=True)
		TokoUtils.copy_file_to_toko(local_multi_py, toko_multi_py)
		return local_batch_path, toko_batch_path, simulation_info

	def process_output(self, local_batch_path, sbatch_output, simulations, toko_batch_path):
		jobid = re.match(r"Submitted batch job (\d+)", sbatch_output.decode('utf-8')).group(1)
		TokoUtils.wait_for_toko_execution(jobid)
		logging.info("Copying output files from toko to local machine...")
		TokoUtils.copy_file_from_toko(toko_batch_path, local_batch_path, is_folder=True)
		for i, simulation in enumerate(simulations):
			# Copy each simulation output to the local machine
			local_sim_folder: Path = Path(simulation.input_file).parent.absolute()
			toko_sim_folder: Path = Path(os.path.join(TOKO_EXECUTION_PATH, local_sim_folder.name))
			TokoUtils.copy_file_from_toko(toko_sim_folder.as_posix(), local_sim_folder.as_posix(), is_folder=True)
			local_lammps_log = os.path.join(local_sim_folder, "log.lammps")
			with open(local_lammps_log, "r") as f:
				lammps_log = f.read()
			self.run_callback(simulation, lammps_log)

	def _get_next_task(self):
		if len(self.queue) == 0:
			return None
		next_task = self.queue[:self.batch_size]
		self.queue = self.queue[self.batch_size:]
		return next_task
