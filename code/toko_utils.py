import logging
import os
import re
import subprocess
from pathlib import Path

import config
from config import LAMMPS_TOKO_EXECUTABLE, TOKO_PARTITION_TO_USE, TOKO_USER, TOKO_URL, TOKO_EXECUTION_PATH
from simulation_task import SimulationTask
from template import TemplateUtils
from utils import confirm, write_local_file


def mkdir_toko(toko_path):
	return run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"mkdir", f"{toko_path}"])


def copy_file_to_toko(local_path: str, toko_path: str, is_folder: bool = False):
	return run_cmd_for_toko(
		lambda user, toko_url:
		["rsync", "-r", local_path + "/", f"{user}@{toko_url}:{toko_path}/"] if is_folder else
		["rsync", local_path, f"{user}@{toko_url}:{toko_path}"]
	)


def copy_file_from_toko(toko_path: str, local_path: str, is_folder: bool = False):
	return run_cmd_for_toko(
		lambda user, toko_url:
		["rsync", "-r", f"{user}@{toko_url}:{toko_path}/", local_path + "/"] if is_folder else
		["rsync", f"{user}@{toko_url}:{toko_path}", local_path]
	)


def run_cmd_for_toko(command_getter):
	command = command_getter(TOKO_USER, TOKO_URL)
	logging.debug(f"Running {command=}")
	return subprocess.check_output(command)


def wait_for_toko_execution(jobid):
	logging.info("Waiting for execution to finish in toko...")
	run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'while [ \"$(/apps/slurm/bin/squeue -hj {jobid})\" != \"\" ]; do sleep 1; done'"])


def submit_toko_script(toko_nano_in: str, toko_sim_folder: str, local_sim_folder: str):
	slurm_code = TemplateUtils.replace_templates(
		TemplateUtils.get_slurm_template(), {
			"lammps_exec": LAMMPS_TOKO_EXECUTABLE,
			"tasks": "1",
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
	copy_file_to_toko(local_slurm_sh, toko_slurm_sh)
	logging.info("Queueing job in toko...")
	return run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'cd {toko_sim_folder}; {config.TOKO_SBATCH} {config.SLURM_SH}'"])


def simulate_in_toko(simulation_task: SimulationTask) -> str:
	input_path = Path(simulation_task.input_file)
	local_sim_folder: Path = input_path.parent.absolute()
	toko_sim_folder: Path = Path(os.path.join(TOKO_EXECUTION_PATH, input_path.parent.name))
	toko_nano_in: Path = Path(os.path.join(toko_sim_folder, input_path.name))
	local_alloy_file: Path = Path(os.path.join(local_sim_folder.parent.parent, "FeCuNi.eam.alloy"))
	toko_alloy_file: Path = Path(os.path.join(toko_sim_folder.parent.parent, "FeCuNi.eam.alloy"))
	logging.info("Creating simulation folder in toko...")
	logging.debug(f"{toko_sim_folder=}")
	logging.debug(f"{local_sim_folder=}")
	logging.debug(f"{toko_nano_in=}")
	logging.debug(f"{local_alloy_file=}")
	logging.debug(f"{toko_alloy_file=}")
	logging.debug("Copying input file...")
	copy_file_to_toko(local_sim_folder.as_posix(), toko_sim_folder.as_posix(), is_folder=True)
	logging.info("Copying alloy files...")
	copy_file_to_toko(local_alloy_file.as_posix(), toko_alloy_file.as_posix())
	sbatch_output = submit_toko_script(toko_nano_in.as_posix(), toko_sim_folder.as_posix(), local_sim_folder.as_posix())
	jobid = re.match(r"Submitted batch job (\d+)", sbatch_output.decode('utf-8')).group(1)
	wait_for_toko_execution(jobid)
	logging.info("Copying output files from toko to local machine...")
	copy_file_from_toko(toko_sim_folder.as_posix(), local_sim_folder.as_posix(), is_folder=True)
	local_lammps_log = os.path.join(local_sim_folder, "log.lammps")
	with open(local_lammps_log, "r") as f:
		lammps_log = f.read()
	return lammps_log
