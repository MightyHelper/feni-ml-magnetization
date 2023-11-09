# Execute nanoparticle simulations in batch
import multiprocessing
import poorly_coded_parser as parser
import nanoparticle
import pandas as pd
from subprocess import CalledProcessError


def execute_all_nanoparticles_in(path, threads, ignore, test=True):
	nanoparticles = parser.load_shapes(path, ignore)
	with multiprocessing.Pool(threads) as p:
		particles = p.starmap(_process_nanoparticle, [(ignore, key, np, test) for key, np in nanoparticles])
	return pd.DataFrame(particles)


def _process_nanoparticle(ignore, key, np, test=True):
	print(f"\033[32m{key}\033[0m")
	if not any([section in key for section in ignore]):
		try:
			np.execute(test_run=test)
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
