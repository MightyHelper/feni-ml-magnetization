# Execute nanoparticle simulations in batch
import multiprocessing
import multiprocessing.dummy
import poorly_coded_parser as parser
import nanoparticle
import pandas as pd
from subprocess import CalledProcessError


def execute_all_nanoparticles_in(path: str, threads: int, ignore: list[str], test: bool=True):
	nanoparticles = parser.load_shapes(path, ignore)
	if threads == 1:
		particles = [_process_nanoparticle(ignore, key, np, test) for key, np in nanoparticles]
	else:
		with multiprocessing.dummy.Pool(threads) as p:
			particles = p.starmap(_process_nanoparticle, [(ignore, key, np, test) for key, np in nanoparticles])
	return pd.DataFrame(particles)


def _process_nanoparticle(ignore: list[str], key: str, np: nanoparticle.Nanoparticle, test: bool=True):
	print(f"\033[32m{key}\033[0m")
	if not any([section in key for section in ignore]):
		try:
			np.execute(test_run=test)
			return parse_ok_execution_results(key, np, test)
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
				"ratio_ni": 0,
				"mag": float('nan')
			}


def parse_ok_execution_results(key: str, np: nanoparticle.Nanoparticle, was_test: bool):
	fe = np.count_atoms_of_type(nanoparticle.FE_ATOM)
	ni = np.count_atoms_of_type(nanoparticle.NI_ATOM)
	# np.plot()
	return {
		"ok": True,
		"key": key,
		"np": np,
		"run_path": np.path,
		"fe": fe,
		"ni": ni,
		"total": fe + ni,
		"ratio_fe": fe / (fe + ni),
		"ratio_ni": ni / (fe + ni),
		"mag": float('nan') if was_test else np.get_magnetism()
	}
