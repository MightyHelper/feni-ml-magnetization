# Execute nanoparticle simulations in batch
import multiprocessing
import poorly_coded_parser as parser
import nanoparticle
import pandas as pd
from subprocess import CalledProcessError


def execute_all_nanoparticles_in(path):
	ignore = [
		# "X-Jannus_Cylinder",
		# "Y-Jannus_Cylinder",
		# "Mix05_PPP-CornerJanus_Cylinder",
		# "Mix10_PPP-CornerJanus_Cylinder",
	]
	nanoparticles = parser.load_shapes(path, ignore)
	with multiprocessing.Pool() as p:
		particles = p.starmap(_process_nanoparticle, [(ignore, key, np) for key, np in nanoparticles.items()])
	df = pd.DataFrame(particles)
	df.drop(columns=["np"], inplace=True)
	print(df.to_string())
	df.to_csv("results.csv")


def _process_nanoparticle(ignore, key, np):
	print(f"\033[32m{key}\033[0m")
	if not any([section in key for section in ignore]):
		try:
			np.execute(test_run=True)
			fe = np.count_atoms_of_type(nanoparticle.FE_ATOM)
			ni = np.count_atoms_of_type(nanoparticle.NI_ATOM)
			return {
				"ok": True,
				"key": key,
				"np": np,
				"run_path": np.path,
				"fe": fe,
				"ni": ni,
				"total": fe + ni,
				"ratio_fe": fe / (fe + ni),
				"ratio_ni": ni / (fe + ni)
			}
		except CalledProcessError:
			print("\033[31mFailed to execute\033[0m")
			return {
				"ok": False,
				"key": key,
				"np": np,
				"run_path": np.path,
				"fe": 0,
				"ni": 0,
				"total": 0,
				"ratio_fe": 0,
				"ratio_ni": 0
			}
