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


def get_running_windows(from_windows: bool = True):
	# wmic.exe process where "name='python.exe'" get commandline, disable stderr

	if from_windows:
		path = "wmic.exe"
	else:
		path = "/mnt/c/Windows/System32/Wbem/wmic.exe"
	result = subprocess.check_output([path, "process", "where", "name='python.exe'", "get", "commandline"], stderr=subprocess.DEVNULL).decode('utf-8').split("\n")
	result = [x.strip() for x in result if x.strip() != ""]
	for execution in {x for result in result if "-in" in result and (x := re.sub(".*?(-in (.*))\n?", "\\2", result).strip()) != ""}:
		folder_name = get_nth_path_element(execution.replace("\\", "/"), -1)
		nano = nanoparticle.Nanoparticle.from_executed(folder_name)
		yield folder_name, nano.run.get_current_step(), nano.title


def get_running_executions():
	if platform.system() == "Windows":
		yield from get_running_windows(True)
	elif platform.system() == "Linux":
		yield from get_running_windows(False)
		yield from get_running_linux()
	else:
		raise Exception(f"Unknown system: {platform.system()}")


def get_running_linux():
	ps_result = os.popen("ps -ef | grep " + config.LAMMPS_EXECUTABLE).readlines()
	for execution in {x for result in ps_result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
		folder_name = get_nth_path_element(execution, -1)
		nano = nanoparticle.Nanoparticle.from_executed(folder_name)
		yield folder_name, nano.run.get_current_step(), nano.title


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


def get_nth_path_element(path: str, n: int) -> str:
	return path.split("/")[n]


def get_path_elements(path: str, f: int, t: int) -> str:
	return "/".join(path.split("/")[f:t])


def dot_dot(path: str):
	return get_path_elements(path, 0, -1)
