import logging
import re
import time
import os

import pandas as pd

import mpilammpsrun as mpilr
import mpilammpswrapper as mpilw
import shapes
import template
import feni_mag
import feni_ovito
import random

FE_ATOM = 1
NI_ATOM = 2

FULL_RUN_DURATION = 300000


def realpath(path):
	return os.path.realpath(path)


class Nanoparticle:
	"""Represents a nanoparticle"""
	regions: list[shapes.Shape]
	atom_manipulation: list[str]
	run: mpilr.MpiLammpsRun
	region_name_map: dict[str, int] = {}
	magnetism: tuple[float, float]
	id: str
	title: str
	path: str
	extra_replacements: dict
	coord: pd.DataFrame
	coord_fe: pd.DataFrame
	coord_ni: pd.DataFrame
	psd_p: pd.DataFrame
	psd: pd.DataFrame
	pec: pd.DataFrame
	total: float
	fe_s: float
	ni_s: float
	fe_c: float
	ni_c: float

	def __init__(self, extra_replacements: dict = None, id_x: str = None):
		self.regions = []
		self.atom_manipulation = []
		self.extra_replacements = {} if extra_replacements is None else extra_replacements
		self.rid = random.randint(0, 10000)
		self.title = self.extra_replacements.get("title", "Nanoparticle")
		self.id = self._gen_identifier() if id_x is None else id_x
		self.path = realpath("../executions/" + self.id) + "/"

	def hardcoded_g_r_crop(self, g_r):
		return g_r[0:60]

	@staticmethod
	def from_executed(path: str):
		"""
		Loads a nanoparticle from an executed simulation
		:param path: Path to the simulation
		:return: A nanoparticle object
		"""
		n = Nanoparticle()
		n.path = realpath(path)
		if not os.path.isdir(n.path):
			raise Exception(f"Path {n.path} is not a directory")
		if not os.path.isfile(n.path + "/log.lammps"):
			raise Exception(f"Path {n.path} does not contain a log.lammps file")
		n.id = n.path.split("/")[-1]
		n.regions = []
		n.atom_manipulation = []
		n.extra_replacements = {}
		n.run = mpilr.MpiLammpsRun.from_path(n.path)
		n.region_name_map = {}
		n.magnetism = n.get_magnetism()
		n.title = n.run.code.split("\n")[0][1:].strip()
		n.coord = n.read_coordination(feni_ovito.COORD_FILENAME)
		n.coord_fe = n.read_coordination(feni_ovito.COORD_FE_FILENAME)
		n.coord_ni = n.read_coordination(feni_ovito.COORD_NI_FILENAME)
		n.psd_p = n.hardcoded_g_r_crop(n.read_psd_p(feni_ovito.PARTIAL_G_R_FILENAME))
		n.psd = n.hardcoded_g_r_crop(n.read_psd(feni_ovito.G_R_FILENAME))
		n.pec = n.read_peh(feni_ovito.PEH_FILENAME)
		surf = n.read_surface_atoms(feni_ovito.SURFACE_FILENAME)
		n.total = surf[0]
		n.fe_s = surf[1]
		n.ni_s = surf[2]
		n.fe_c = surf[3]
		n.ni_c = surf[4]
		return n

	def columns_for_dataset(self):
		# col_order = ["psd", "psd11", "psd12", "psd22", "coordc", "coordc_fe", "coordc_ni", "pec", "fe_s", "ni_s", "fe_c", "ni_c", "n_fe", "n_ni", "tmg"]
		out = [
			self._drop_index(pd.DataFrame([self.get_descriptor_name()], columns=["name"])),
		]
		for k, (v, count) in {
			'psd': (self.psd[['psd']].copy(), 60),
			'psd11': (self.psd_p[['1-1']].copy(), 60),
			'psd12': (self.psd_p[['1-2']].copy(), 60),
			'psd22': (self.psd_p[['2-2']].copy(), 60),
			'coordc': (self.coord[['count']].copy(), 100),
			'coordc_fe': (self.coord_fe[['count']].copy(), 100),
			'coordc_ni': (self.coord_ni[['count']].copy(), 100),
			'pec': (self.pec[['count']].copy(), 100),
		}.items():
			out += [self._get_pivoted_df(v, k, expected_row_count=count)]
		for k, v in {
			'fe_s': self._drop_index(pd.DataFrame([self.fe_s], columns=["fe_s"])),
			'ni_s': self._drop_index(pd.DataFrame([self.ni_s], columns=["ni_s"])),
			'fe_c': self._drop_index(pd.DataFrame([self.fe_c], columns=["fe_c"])),
			'ni_c': self._drop_index(pd.DataFrame([self.ni_c], columns=["ni_c"])),
			'n_fe': self._drop_index(pd.DataFrame([self.count_atoms_of_type(FE_ATOM)], columns=["n_fe"])),
			'n_ni': self._drop_index(pd.DataFrame([self.count_atoms_of_type(NI_ATOM)], columns=["n_ni"])),
			'tmg': self._drop_index(pd.DataFrame([self.magnetism[0]], columns=["tmg"])),
		}.items():
			out += [v]

		# Concat columns of all dataframes
		return pd.concat(out, axis=1)

	def _drop_index(self, df):
		df.index = ["" for _ in df.index]
		return df

	def _get_pivoted_df(self, df, name, expected_row_count=100):
		row_count = df.shape[0]
		if row_count > expected_row_count:
			logging.debug("Expanding " + name)
			df["index"] = df.index // (row_count / expected_row_count)
			df = df.groupby("index").mean()
			df.reset_index(inplace=True)
		if row_count < expected_row_count:
			logging.debug("Filling " + name)
			# Expand the range filling the gaps with 0
			new_df = df.reindex(range(expected_row_count), fill_value=0)
			for i in range(row_count):
				new_df.iloc[i] = 0
				new_df.iloc[int((float(expected_row_count) / float(row_count)) * i)] = df.iloc[i]
			df = new_df
		df = df.transpose()
		df.columns = [f"{name}_{int(i) + 1}" for i in df.columns]
		df.index = ["" for _ in df.index]
		return df

	def read_coordination(self, filename):
		try:
			df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
			df = df.iloc[:, 0:2]
			df.columns = ["coordination", "count"]
			df = df.astype({"coordination": float, "count": float})
			df["coordination"] -= 0.5
		except FileNotFoundError:
			df = pd.DataFrame(columns=["coordination", "count"])
		return df

	def read_psd_p(self, filename):
		try:
			df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
			df.columns = ["radius", "1-1", "1-2", "2-2"]
			df = df.astype({"radius": float, "1-1": float, "1-2": float, "2-2": float})
		except FileNotFoundError:
			df = pd.DataFrame(columns=["radius", "1-1", "1-2", "2-2"])
		return df

	def read_psd(self, filename):
		try:
			df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
			df.columns = ["radius", "psd"]
			df = df.astype({"radius": float, "psd": float})
			if len(df) > 100:
				df["index"] = df.index // (len(df) / 100)
				df = df.groupby("index").mean()
				df.reset_index(inplace=True)
				df.drop(columns=["index"], inplace=True)
		except FileNotFoundError:
			df = pd.DataFrame(columns=["radius", "psd"])
		return df

	def read_surface_atoms(self, filename):
		try:
			with open(self.path + "/" + filename, "r") as f:
				line = f.readlines()[1].strip()
				return [float(x) for x in re.split(" +", line)]
		except FileNotFoundError:
			df = pd.DataFrame(columns=["type", "count"])
		return df

	def read_peh(self, filename):
		try:
			df = pd.read_csv(self.path + "/" + filename, delimiter=" ", skiprows=2, header=None)
			df = df.iloc[:, 0:2]
			df.columns = ["pe", "count"]
			df = df.astype({"pe": float, "count": float})
		except FileNotFoundError:
			df = pd.DataFrame(columns=["pe", "count"])
		return df

	def execute(self, test_run: bool = True, **kwargs):
		"""
		Executes the nanoparticle simulation
		:param test_run:  If true, only one dump will be generated
		:param kwargs: 	Extra arguments to pass to the lammps run
		:return:
		"""
		code = self._build_lammps_code(test_run)
		os.mkdir(self.path)  # Create the run directory
		dumps, lammps_run = self._build_lammps_run(code, kwargs, test_run)
		self.run = lammps_run.execute()
		if not test_run:
			feni_mag.extract_magnetism(self.path + "/log.lammps", out_mag=self.path + "/magnetism.txt", digits=4)
			feni_ovito.parse(filenames={'base_path': self.path, 'dump': dumps[-1]})
			self.magnetism = self.get_magnetism()

	def _build_lammps_run(self, code, kwargs, test_run):
		lammps_run = mpilr.MpiLammpsRun(
			code, {
				"omp": mpilw.OMPOpt(use=False, n_threads=2),
				"mpi": mpilw.MPIOpt(use=False, hw_threads=False, n_threads=4),
				"cwd": self.path,
				**kwargs
			},
			dumps := [f"iron.{i}.dump" for i in ([0] if test_run else [*range(0, FULL_RUN_DURATION + 1, 100000)])],
			file_name=self.path + "nanoparticle.in"
		)
		return dumps, lammps_run

	def _build_lammps_code(self, test_run):
		return template.replace_templates(template.get_lammps_template(), {
			"region": self.get_region(),
			"run_steps": str(0 if test_run else FULL_RUN_DURATION),
			"title": f"{self.title}",
			**self.extra_replacements
		})

	def _gen_identifier(self):
		return f"simulation_{time.time()}_{self.rid}"

	def get_magnetism(self):
		try:
			with open(self.path + "/magnetism.txt", "r") as f:
				lines = f.readlines()
			result = [float(x) for x in lines[1].split(" ")]
			return result[0], result[1]
		except FileNotFoundError:
			return float('nan'), float('nan')

	def get_region(self):
		out = ""
		for i in self.atom_manipulation:
			if isinstance(i, str):
				out += i + "\n"
			elif isinstance(i, shapes.Shape):
				out += i.get_region(f"reg{self.regions.index(i)}") + "\n"
				raise Exception("Shapes are supported but not allowed")
			else:
				raise Exception(f"Unknown type: {type(i)}")
		return out

	def plot(self):
		if self.run is None:
			raise Exception("Run not executed")
		self.run.dumps[0].plot()

	def count_atoms_of_type(self, atom_type, dump_idx=0):
		return self.run.dumps[dump_idx].count_atoms_of_type(atom_type)

	def total_atoms(self, dump_idx=0):
		return self.run.dumps[dump_idx].dump['number_of_atoms']

	def __str__(self):
		return f"Nanoparticle(regions={len(self.regions)} items, atom_manipulation={len(self.atom_manipulation)} items, title={self.title})"

	def __repr__(self):
		return self.__str__()

	def get_descriptor_name(self):
		return "/".join(self.title.replace("../Shapes", "Descriptors").split("/")[:-1])

	def get_full_coord(self):
		out = self.coord.copy()
		out["count_fe"] = self.coord_fe["count"]
		out["count_ni"] = self.coord_ni["count"]
		return out
