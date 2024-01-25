import asyncio
import base64
import os
import random
import re
from pathlib import Path
from typing import TypeVar, Callable, Any, cast

from asyncssh import SSHClientConnection


def get_current_step(lammps_log):
    """
    Get the current step of a lammps log file
    """
    step = -1
    try:
        with open(lammps_log, "r") as f:
            lines = f.readlines()
            # noinspection PyBroadException
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


def parse_nanoparticle_name(key: str) -> tuple[str, str, str, str, str]:
    shape, distribution, interface, pores, index = None, None, None, None, None
    # noinspection PyBroadException
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


def get_path_elements(path: str, f: int, t: int) -> str:
    return "/".join(path.split("/")[f:t])


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


def write_local_file(path: Path | str, content: str):
    with open(path, "w") as f:
        f.write(content)


def read_local_file(path: Path | str) -> str | None:
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


def get_matching(distributions: dict[str, T], processed_name: str, error: str) -> T:
    for key, value in distributions.items():
        if key in processed_name:
            return value
    raise Exception(error)


def assign_nanoparticle_name(name: str) -> dict[str, int | str]:
    shape, distribution, interface, pores, index = parse_nanoparticle_name(name)
    dist = distribution.split(".")
    intf = interface.split(".")
    prs = pores.split(".")
    out = {
        'Shape': shape,
        'Distribution': dist[0],
        'Distribution_full': distribution,
        'Distribution_data': "" if len(dist) == 1 else ".".join(dist[1:]),
        'Interface': intf[0],
        'Interface_full': interface,
        'Interface_data': "" if len(intf) == 1 else ".".join(intf[1:]),
        'Pores': prs[0],
        'Pores_full': pores,
        'Pores_data': "" if len(prs) == 1 else ".".join(prs[1:]),
        'Index': int(index)
    }
    return out


def set_type(typ: type[T], item: Any) -> T:
    assert isinstance(typ, type), f"typ must be a type, got {typ}"
    assert isinstance(item, typ), f"item must be of type {typ}, got {item} ({type(item)})"
    return cast(typ, item)


def assert_type(typ: type[T], item: T) -> T:
    assert isinstance(typ, type), f"typ must be a type, got {typ}"
    assert isinstance(item, typ), f"item must be of type {typ}, got {item} ({type(item)})"
    return item

def ssh_task(func):
    def wrapper(conn: SSHClientConnection, *args, **kwargs) -> asyncio.Task:
        return asyncio.create_task(conn.run(func(*args, **kwargs)))
    return wrapper