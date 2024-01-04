import base64
import logging
import os
import platform
import random
import re
from pathlib import Path
from typing import TypeVar, Callable

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
    highlights = [
        r"(^(?P<zero>0+(.0+)))|([^.\d](?P<zero_1>0+(.0+))$)|(^(?P<zero_2>0+(.0+))$)|([^.\d](?P<zero_3>0+(.0+))[^.\d])"]


def parse_nanoparticle_name(key) -> tuple[str, str, str, str, str]:
    shape, distribution, interface, pores, index = None, None, None, None, None
    try:
        filename = os.path.basename(key)
        if filename.endswith(".in"):
            filename = filename[:-3]
        parts = filename.split("_")
        shape = parts[0]
        distribution = parts[1]
        interface = parts[2]
        pores = parts[3]
        index = parts[4]
    except Exception:
        pass
    return shape, distribution, interface, pores, index


def add_task(folder, progress: Progress, step, tasks, title):
    logging.info(f"Found running execution: {folder} ({step})")
    tasks[folder] = progress.add_task(f"{os.path.basename(folder)} ({title})", total=None if step == -1 else config.FULL_RUN_DURATION)


def get_path_elements(path: str, f: int, t: int) -> str:
    return "/".join(path.split("/")[f:t])


def dot_dot(path: str):
    return get_path_elements(path, 0, -1)


def resolve_path(path: Path) -> str:
    """
    Returns the absolute path to the file or directory
    :param path: path to resolve
    :return: absolute path
    """
    return path.absolute().as_posix()


def confirm(message):
    res = input(f"{message} (y/n) ")
    if res != "y":
        raise ValueError("Oki :c")


def get_file_name(input_file):
    return "/".join(input_file.split("/")[-2:])


def get_index(lines: list[str], section: str, head: str) -> int | None:
    """
    Get the index of a section in a list of lines
    :param lines:
    :param section:
    :param head:
    :return:
    """
    try:
        return [i for i, l in enumerate(lines) if l.startswith(head + section)][0]
    except IndexError:
        return None


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


def read_local_file(path) -> str | None:
    try:
        with open(path, "r") as template:
            return template.read()
    except FileNotFoundError:
        return None


T = TypeVar("T")
V = TypeVar("V")


def opt(value: T | None, func: Callable[[T | None], V]) -> V:
    return None if value is None else func(value)


def column_values_as_float(line):
    return [float(x) for x in line.split(" ") if x != ""]
