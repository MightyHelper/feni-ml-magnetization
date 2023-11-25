import re
from io import StringIO

import pandas as pd

from config import LOCAL_EXECUTION_PATH

DATA_START_PATTERN = re.compile("Step\\s+Temp")
DATA_END_PATTERN = re.compile("Loop time")


def extract_magnetism(in_log='log.lammps', out_mag='Magnetization.txt', digits=2):
	with open(in_log, "r") as in_file:
		lines = in_file.readlines()
	pd_df = pd.read_table(StringIO("".join(get_relevant_lines(lines))), sep="\\s+")
	pd_df.set_index('Step', inplace=True)
	last_step_name = pd_df.iloc[-1].name
	new_index = last_step_name - 100_000
	mag_stats = ' '.join(pd_df.loc[new_index:last_step_name]['v_magnorm'].agg(['mean', 'std']).to_numpy().round(digits).astype(str))
	with open(out_mag, "w") as out_file:
		out_file.write("# TotalMag Error\n" + mag_stats)


def get_relevant_lines(lines):
	start_index = [i for i, line in enumerate(lines) if DATA_START_PATTERN.search(line)]
	end_index = [i for i, line in enumerate(lines) if DATA_END_PATTERN.search(line)]
	if len(start_index) == 0 or len(end_index) == 0:
		raise Exception("Couldn't find pattern")
	start_index = start_index[0]
	end_index = end_index[0]
	relevant_lines = lines[start_index:end_index]
	return relevant_lines


if __name__ == '__main__':
	extract_magnetism(LOCAL_EXECUTION_PATH + "/log.lammps")
