import re
from io import StringIO
from pathlib import Path

import pandas as pd

from config.config import LOCAL_EXECUTION_PATH
from lammps.lammpsdump import LammpsLog

LAST_N_MAGNETISM_AVG = 600_000


class MagnetismExtractor:
    @staticmethod
    def extract_magnetism(in_log='log.lammps', out_mag='Magnetization.txt', digits=2):
        pd_df = LammpsLog(Path(in_log)).log
        last_step_name = pd_df.iloc[-1].name
        new_index = last_step_name - LAST_N_MAGNETISM_AVG
        mag_stats = ' '.join(pd_df.loc[new_index:last_step_name]['v_magnorm'].agg(['mean', 'std']).to_numpy().round(digits).astype(str))
        with open(out_mag, "w") as out_file:
            out_file.write("# TotalMag Error\n" + mag_stats)


if __name__ == '__main__':
    MagnetismExtractor.extract_magnetism(LOCAL_EXECUTION_PATH + "/log.lammps")
