# Execute nanoparticle simulations in batch
import logging
import multiprocessing
import multiprocessing.dummy
import random
from typing import cast

import config
import nanoparticlebuilder
import poorly_coded_parser as parser
import nanoparticle
import pandas as pd
from subprocess import CalledProcessError


def execute_all_nanoparticles_in(path: str, threads: int, ignore: list[str], test: bool = True, seed_count: int = 1, seed: int = 123, count_only: bool = False):
	nanoparticles = build_nanoparticles_to_execute(ignore, path, seed, seed_count)
	if count_only:
		return len(nanoparticles)
	if threads == 1:
		particles = [_process_nanoparticle(ignore, key, np, test) for key, np in nanoparticles]
	else:
		with multiprocessing.dummy.Pool(threads) as p:
			particles = p.starmap(_process_nanoparticle, [(ignore, key, np, test) for key, np in nanoparticles])
	return pd.DataFrame(particles)


def build_nanoparticles_to_execute(ignore, path, seed, seed_count):
	nano_builders = parser.load_shapes(path, ignore)
	nanoparticles = []
	random.seed(seed)
	for key, nano in nano_builders:
		nano = cast(nanoparticlebuilder.NanoparticleBuilder, nano)
		if nano.is_random():
			for i in range(seed_count):
				seeds = [random.randint(0, 100000) for _ in range(len(nano.seed_values))]
				logging.info(f"Using seeds {seeds}")
				nanoparticles.append((key, nano.build(seeds)))
		else:
			nanoparticles.append((key, nano.build()))
	return nanoparticles


def _process_nanoparticle(ignore: list[str], key: str, np: nanoparticle.Nanoparticle, test: bool = True):
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
	fe = np.count_atoms_of_type(config.FE_ATOM)
	ni = np.count_atoms_of_type(config.NI_ATOM)
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
