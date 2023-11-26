import datetime
import logging
import os
import platform
import re
import subprocess

from rich.highlighter import RegexHighlighter
from rich.progress import Progress

import config
import nanoparticle


def parse_execution_info(folder):
	out = {
		'real_date': None,
		'title': None,
		'mag': None,
		'toko': False
	}
	if "_" in folder:
		parts = folder.split("_")
		sim, date = parts[0], parts[1]
		out['real_date'] = datetime.datetime.utcfromtimestamp(float(date))
	out['title'] = get_execution_title(folder)
	out['mag'] = get_magnetism(folder)
	out['toko'] = "slurm.sh" in os.listdir(config.LOCAL_EXECUTION_PATH + "/" + folder)
	return out


def get_execution_title(folder):
	try:
		with open(config.LOCAL_EXECUTION_PATH + "/" + folder + "/nanoparticle.in", "r") as f:
			lines = f.readlines()
			return lines[0][2:].strip()
	except FileNotFoundError:
		pass
	return "Unknown"


def get_magnetism(folder):
	try:
		with open(config.LOCAL_EXECUTION_PATH + "/" + folder + "/magnetism.txt", "r") as f:
			lines = f.readlines()
			return lines[1].strip()
	except FileNotFoundError:
		pass
	return "Unknown"


def get_current_step(lammps_log):
	"""
	Get the current step of a lammps log file
	"""
	step = -1
	try:
		with open(lammps_log, "r") as f:
			lines = f.readlines()
			try:
				split = re.split(r" +", lines[-1].strip())
				step = int(split[0])
			except Exception:
				pass
	except FileNotFoundError:
		pass
	return step


def get_title(path):
	title = "Unknown"
	try:
		with open(path, "r") as f:
			lines = f.readlines()
			title = lines[0][2:].strip()
	except FileNotFoundError:
		pass
	return title


def get_runninng_windows(from_windows: bool = True):
	# wmic.exe process where "name='python.exe'" get commandline, disable stderr

	if from_windows:
		path = "wmic.exe"
	else:
		path = "/mnt/c/Windows/System32/Wbem/wmic.exe"
	result = subprocess.check_output([path, "process", "where", "name='python.exe'", "get", "commandline"], stderr=subprocess.DEVNULL).decode('utf-8').split("\n")
	result = [x.strip() for x in result if x.strip() != ""]
	for execution in {x for result in result if "-in" in result and (x := re.sub(".*?(-in (.*))\n?", "\\2", result).strip()) != ""}:
		execution = execution.replace("\\", "/")
		parts = execution.split("/")
		foldername = '/'.join(parts[:-1])
		step = get_current_step(foldername + "/log.lammps")
		title = get_title(foldername + "/nanoparticle.in")
		yield foldername, step, title


def get_running_executions():
	if platform.system() == "Windows":
		yield from get_runninng_windows(True)
	elif platform.system() == "Linux":
		yield from get_runninng_windows(False)
		yield from get_runninng_linux()
	else:
		raise Exception(f"Unknown system: {platform.system()}")


def get_runninng_linux():
	ps_result = os.popen("ps -ef | grep " + config.LAMMPS_EXECUTABLE).readlines()
	for execution in {x for result in ps_result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
		parts = execution.split("/")
		foldername = '/'.join(parts[:-1])
		step = get_current_step(foldername + "/log.lammps")
		title = get_title(foldername + "/nanoparticle.in")
		yield foldername, step, title


def filter_empty(l: list) -> list:
	return [x for x in l if x != ""]


class ZeroHighlighter(RegexHighlighter):
	"""Apply style to anything that looks like non zero."""
	base_style = "zero."
	highlights = [r"(^(?P<zero>0+(.0+)))|([^.\d](?P<zero_1>0+(.0+))$)|(^(?P<zero_2>0+(.0+))$)|([^.\d](?P<zero_3>0+(.0+))[^.\d])"]


def parse_nanoparticle_name(key):
	parts = key.split("/")
	ptype = parts[2]
	subtype = parts[3]
	subsubtype = parts[4] if not parts[4].endswith(".in") else ""
	return ptype, subtype, subsubtype


def add_task(folder, progress: Progress, step, tasks, title):
	logging.info(f"Found running execution: {folder} ({step})")
	tasks[folder] = progress.add_task(f"{folder} ({title})", total=None if step == -1 else nanoparticle.FULL_RUN_DURATION)
