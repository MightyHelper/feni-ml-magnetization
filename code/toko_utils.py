import logging
import re
import subprocess

import config
from config import LAMMPS_TOKO_EXECUTABLE, TOKO_PARTITION_TO_USE, USER, TOKO_URL
from template import TemplateUtils
from utils import confirm, get_file_name


def mkdir_toko(toko_path):
	return run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"mkdir", f"{toko_path}"])


def copy_file_to_toko(local_path, toko_path):
	return run_cmd_for_toko(lambda user, toko_url: ["scp", local_path, f"{user}@{toko_url}:{toko_path}"])


def copy_file_from_toko(toko_path, local_path):
	return run_cmd_for_toko(lambda user, toko_url: ["scp", "-r", f"{user}@{toko_url}:{toko_path}", local_path])


def run_cmd_for_toko(command_getter):
	command = command_getter(USER, TOKO_URL)
	logging.debug(f"Running {command=}")
	return subprocess.check_output(command)


def wait_for_toko_execution(jobid):
	logging.info("Waiting for execution to finish in toko...")
	run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'while [ \"$(/apps/slurm/bin/squeue -hj {jobid})\" != \"\" ]; do sleep 1; done'"])


def write_toko_script(toko_nano_in, toko_sim_folder, local_sim_folder):
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
	with open(local_sim_folder + "/" + config.SLURM_SH, "w") as f:
		f.write(slurm_code)
	logging.info(f"Copying {config.SLURM_SH} to toko...")
	copy_file_to_toko(local_sim_folder + "/" + config.SLURM_SH, toko_sim_folder + "/" + config.SLURM_SH)
	logging.info("Queueing job in toko...")
	return run_cmd_for_toko(lambda user, toko_url: ["ssh", f"{user}@{toko_url}", f"sh -c 'cd {toko_sim_folder}; /apps/slurm/bin/sbatch {config.SLURM_SH}'"])


def simulate_in_toko(input_file):
	confirm("Run in TOKO?")
	local_sim_folder = "/".join(input_file.split("/")[:-1])
	toko_sim_folder = "~/scratch/projects/magnetism/simulations/" + "/".join(get_file_name(input_file).split("/")[:-1])
	toko_nano_in = "~/scratch/projects/magnetism/simulations/" + get_file_name(input_file)
	logging.info("Creating simulation folder in toko...")
	mkdir_toko(toko_sim_folder)
	logging.info("Copying alloy files...")
	copy_file_to_toko("/".join(input_file.split("/")[:-3]) + "/FeCuNi.eam.alloy", "~/scratch/projects/magnetism/FeCuNi.eam.alloy")
	logging.info("Copying input file...")
	copy_file_to_toko(input_file, toko_nano_in)
	code = write_toko_script(toko_nano_in, toko_sim_folder, local_sim_folder)
	jobid = re.match(r"Submitted batch job (\d+)", code.decode('utf-8')).group(1)
	wait_for_toko_execution(jobid)
	logging.info("Copying output files from toko to local machine...")
	copy_file_from_toko(toko_sim_folder, "/".join(local_sim_folder.split("/")[:-1]))
