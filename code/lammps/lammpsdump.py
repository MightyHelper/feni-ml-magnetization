import logging
from functools import cached_property, cache
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from utils import get_index, opt, column_values_as_float, read_local_file, write_local_file

DATA_START_PATTERN = re.compile("Step\\s+Temp")
DATA_END_PATTERN = re.compile("Loop time")
LAST_N_MAGNETISM_AVG = 600_000


class LammpsDump:
    """
    Functions to parse a dump file
    """
    path: Path

    def __init__(self, path: Path):
        self.path = path

    @cached_property
    def _content(self) -> list[str]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dump file {self.path} does not exist!")
        return read_local_file(self.path).strip().split("\n")

    @cached_property
    def dump(self) -> dict[str, any]:
        try:
            return {
                "timestep": opt(
                    get_index(self._content, "TIMESTEP", "ITEM: "),
                    lambda x: int(self._content[x + 1])
                ),
                "number_of_atoms": opt(
                    get_index(self._content, "NUMBER OF ATOMS", "ITEM: "),
                    lambda x: int(self._content[x + 1])
                ),
                "box_bounds": opt(
                    get_index(self._content, "BOX BOUNDS", "ITEM: "),
                    lambda o: np.array([
                        column_values_as_float(line)
                        for line in self._content[o + 1:o + 4]
                    ])
                ),
                "atoms": opt(
                    get_index(self._content, "ATOMS", "ITEM: "),
                    lambda o: np.array([
                        column_values_as_float(line)
                        for line in self._content[o + 1:]
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

    @cache
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
        try:
            df: pd.DataFrame = pd.read_table(StringIO("\n".join(self._content[self._df_start:self._df_end])), sep="\\s+")
            df.set_index('Step', inplace=True)
            return df
        except ValueError as e:
            raise Exception(f"Could not parse log file {self.path}!") from e

    @cached_property
    def _exec_info(self) -> dict[str, int | float]:
        match = re.search("Loop time of (?P<time>[\d.]+) on (?P<procs>\d+) procs for (?P<steps>\d+) steps with (?P<atoms>\d+) atoms", self._content[self._df_end])
        return {
            "time": float(match.group("time")),
            "procs": int(match.group("procs")),
            "steps": int(match.group("steps")),
            "atoms": int(match.group("atoms"))
        }

    @property
    def exec_time(self) -> float:
        return self._exec_info['time']

    @property
    def exec_procs(self) -> int:
        return self._exec_info['procs']

    @property
    def exec_steps(self) -> int:
        return self._exec_info['steps']

    @property
    def exec_atoms(self) -> int:
        return self._exec_info['atoms']

    @cached_property
    def _content(self) -> list[str]:
        if not self.path.exists():
            raise FileNotFoundError(f"Log file {self.path} does not exist!")
        return read_local_file(self.path).strip().split("\n")

    @cached_property
    def step_count(self) -> int:
        return self.log.iloc[-1].name

    @cached_property
    def magnetism(self) -> pd.Series:
        new_index: int = self.step_count - LAST_N_MAGNETISM_AVG
        return self.log.loc[new_index:self.step_count]['v_magnorm'].agg(['mean', 'std'])

    @cached_property
    def total_energy(self) -> pd.Series:
        new_index: int = self.step_count - LAST_N_MAGNETISM_AVG
        return (self.log.loc[new_index:self.step_count]['TotEng'] / self.exec_atoms).agg(['mean', 'std'])

    def save_mag_to_file(self, out_mag: Path, digits=2):
        mag_stats: str = f"{self.magnetism['mean']:.{digits}f} {self.magnetism['std']:.{digits}f}"
        write_local_file(out_mag, f"# TotalMag Error\n{mag_stats}")
        logging.info(f"Saved magnetization for {self.path} to {out_mag}")

    def plot_tmg(self, title: str = "Total Magnetization"):
        self.log.plot(y='v_magnorm', title=title)
        plt.show()

    def plot_tmg_vs_teng(self, title):
        # Set color to index
        plt.scatter(self.log['TotEng'], self.log['v_magnorm'], c=self.log.index, cmap='viridis', alpha=0.5, s=2)
        plt.plot(self.log['TotEng'], self.log['v_magnorm'], color='gray', alpha=0.5)  # Connect points
        plt.xlabel('Total energy')
        plt.ylabel('Magnetization')
        plt.title(title)
        plt.colorbar(label='Timestep')
        plt.grid(True)
        plt.show()

    def plot_teng_hist(self, title: str = "Total Energy Histogram"):
        # percentile_5 = self.log['TotEng'].quantile(0.05)
        # filtered: pd.DataFrame = self.log[self.log['TotEng'] > percentile_5]
        # filtered['TotEng'].plot.hist(title=title, bins=40)
        self.log.plot(y="TotEng", title=title, alpha=0.5)
        plt.show()

    @cached_property
    def _df_start(self):
        return self.index_line(DATA_START_PATTERN)

    @cached_property
    def _df_end(self):
        return self.index_line(DATA_END_PATTERN)

    def index_line(self, pattern: re.Pattern) -> int:
        index = [i for i, line in enumerate(self._content) if pattern.search(line)]
        if len(index) == 0:
            raise Exception("Couldn't find pattern")
        if len(index) > 1:
            logging.debug(f"Found multiple patterns {index}")
        return index[0]


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
