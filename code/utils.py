import base64
import logging
import os
import platform
import random
import re

from rich.highlighter import RegexHighlighter
from rich.progress import Progress

import config


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
	subsubtype = re.sub("[-_]?" + subtype, "", subsubtype, flags=re.IGNORECASE)
	subtype = re.sub("[-_]?" + ptype, "", subtype, flags=re.IGNORECASE)
	subsubtype = re.sub("[-_]?" + subtype, "", subsubtype, flags=re.IGNORECASE)
	subsubtype = re.sub("[-_]?" + ptype, "", subsubtype, flags=re.IGNORECASE)
	return ptype, subtype, subsubtype


def add_task(folder, progress: Progress, step, tasks, title):
	logging.info(f"Found running execution: {folder} ({step})")
	tasks[folder] = progress.add_task(f"{folder} ({title})", total=None if step == -1 else config.FULL_RUN_DURATION)


def get_path_elements(path: str, f: int, t: int) -> str:
	return "/".join(path.split("/")[f:t])


def dot_dot(path: str):
	return get_path_elements(path, 0, -1)


def resolve_path(path):
	return path.absolute().as_posix()


def confirm(message):
	res = input(f"{message} (y/n) ")
	if res != "y":
		raise ValueError("Oki :c")


def get_file_name(input_file):
	return "/".join(input_file.split("/")[-2:])


def get_index(lines: list[str], section: str, head: str) -> int:
	"""
	Get the index of a section in a list of lines
	:param lines:
	:param section:
	:param head:
	:return:
	"""
	return [i for i, l in enumerate(lines) if l.startswith(head + section)][0]


def generate_random_filename():
	return base64.b32encode(random.randbytes(5)).decode("ascii")


def drop_index(df):
	df.index = ["" for _ in df.index]
	return df


def realpath(path):
	return os.path.realpath(path)


def write_local_file(path, slurm_code):
	with open(path, "w") as f:
		f.write(slurm_code)


def read_local_file(path):
	template = open(path, "r")
	return template.read()
