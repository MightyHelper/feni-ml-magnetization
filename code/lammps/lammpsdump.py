import logging
from functools import cached_property
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import utils
from rich import print as rprint
from utils import get_index, opt, column_values_as_float

DATA_START_PATTERN = re.compile("Step\\s+Temp")
DATA_END_PATTERN = re.compile("Loop time")


class LammpsDump:
    """
    Functions to parse a dump file
    """
    path: Path

    def __init__(self, path: Path):
        self.path = path

    @cached_property
    def dump(self) -> dict[str, any]:
        return self._parse()

    def _parse(self) -> dict[str, any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dump file {self.path} does not exist!")
        lines = utils.read_local_file(self.path).strip().split("\n")
        try:
            return {
                "timestep": opt(
                    get_index(lines, "TIMESTEP", "ITEM: "),
                    lambda x: int(lines[x + 1])
                ),
                "number_of_atoms": opt(
                    get_index(lines, "NUMBER OF ATOMS", "ITEM: "),
                    lambda x: int(lines[x + 1])
                ),
                "box_bounds": opt(
                    get_index(lines, "BOX BOUNDS", "ITEM: "),
                    lambda o: np.array([
                        column_values_as_float(line)
                        for line in lines[o + 1:o + 4]
                    ])
                ),
                "atoms": opt(
                    get_index(lines, "ATOMS", "ITEM: "),
                    lambda o: np.array([
                        column_values_as_float(line)
                        for line in lines[o + 1:]
                    ])
                )
            }
        except ValueError as e:
            raise Exception(f"Could not parse dump file {self.path}!") from e

    def plot(self):
        t = self.dump['atoms'][:, DUMP_ATOM_TYPE]
        x = self.dump['atoms'][:, DUMP_ATOM_X]
        y = self.dump['atoms'][:, DUMP_ATOM_Y]
        z = self.dump['atoms'][:, DUMP_ATOM_Z]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # noinspection SpellCheckingInspection
        ax.scatter(x, y, z, c=t, marker='.', cmap='coolwarm')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(self.dump['box_bounds'][0][0], self.dump['box_bounds'][0][1])
        ax.set_ylim(self.dump['box_bounds'][1][0], self.dump['box_bounds'][1][1])
        ax.set_zlim(self.dump['box_bounds'][2][0], self.dump['box_bounds'][2][1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def count_atoms_of_type(self, atom_type):
        return np.count_nonzero(self.dump['atoms'][:, DUMP_ATOM_TYPE] == atom_type)

    def __str__(self):
        return f"<Dump@{self.dump['timestep']} for {self.path}>"

    def __repr__(self):
        return str(self)


class LammpsLog:
    path: Path

    def __init__(self, path: Path):
        self.path = path

    @cached_property
    def log(self) -> pd.DataFrame:
        return self._parse()

    def _parse(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Log file {self.path} does not exist!")
        lines = utils.read_local_file(self.path).strip().split("\n")
        try:
            df: pd.DataFrame = pd.read_table(StringIO("\n".join(LammpsLog.__get_relevant_lines(lines))), sep="\\s+")
            df.set_index('Step', inplace=True)
            return df
        except ValueError as e:
            raise Exception(f"Could not parse log file {self.path}!") from e

    @staticmethod
    def __get_relevant_lines(lines):
        start_index = [i for i, line in enumerate(lines) if DATA_START_PATTERN.search(line)]
        end_index = [i for i, line in enumerate(lines) if DATA_END_PATTERN.search(line)]
        if len(start_index) == 0 or len(end_index) == 0:
            raise Exception("Couldn't find pattern")
        if len(start_index) > 1 or len(end_index) > 1:
            logging.warning(f"Found multiple patterns {start_index} {end_index}")
        return lines[start_index[0]:end_index[0]]

    def plot_tmg(self, title: str = "Total Magnetization"):
        self.log.plot(y='v_magnorm', title=title)
        plt.show()


DUMP_ATOM_TYPE = 0
DUMP_ATOM_ID = 1
DUMP_ATOM_X = 2
DUMP_ATOM_Y = 3
DUMP_ATOM_Z = 4
DUMP_ATOM_VX = 5
DUMP_ATOM_VY = 6
DUMP_ATOM_VZ = 7
DUMP_ATOM_C1 = 8
DUMP_ATOM_C2 = 9
DUMP_ATOM_C3 = 10
DUMP_ATOM_PE = 11
DUMP_ATOM_KE = 12
