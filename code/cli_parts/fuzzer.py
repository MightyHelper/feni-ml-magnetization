import logging
import re
from pathlib import Path

from bayes_opt import BayesianOptimization, UtilityFunction
import typer
from rich import print as rprint

import config
import executor
import poorly_coded_parser as parser
import utils

fuzzer = typer.Typer(add_completion=False, no_args_is_help=True, name="fuzzer")


def get_full_function(path: str):
	with open(path, 'r') as f:
		contents = f.read()
	# Find all {{x}}
	keys = re.findall(r"{{(.*?)}}", contents)

	def fuzz(**kwargs):
		kwargs = {k: str(v) for k, v in kwargs.items()}
		rprint(kwargs)
		_, nanoparticle = parser.parse_single_shape(path, False, replacements=kwargs)
		nanoparticle = nanoparticle.build(title='Fuzzing ' + nanoparticle.title)
		nanoparticle.execute(True)
		result = executor.parse_ok_execution_results("fuzzed", nanoparticle, True)
		atom_count = result['total']
		ratio = result['ratio_ni']
		return atom_count, ratio, nanoparticle

	return keys, fuzz


@fuzzer.command()
def bayes(
	path: Path,
	target_atom_count: int = 1250,
	target_ratio: float = 0.33,
	plot: bool = False
):
	"""
	Use bayesian search to find the correct value for a parameter in a nanoparticle
	:param path: The path to the nanoparticle
	:param target_atom_count: The target atom count
	:param target_ratio: The target ratio
	:param plot: Whether to plot the nanoparticle
	"""
	rprint(f"Using bayesian search to find the correct value for a parameter in a nanoparticle")
	path = utils.resolve_path(path)
	keys, run_fuzzer = get_full_function(path)
	target_values = [target_atom_count, target_ratio]

	def result(**kwargs):
		atom_count, ratio, nanoparticle = run_fuzzer(**kwargs)
		return -sum((actual - target) ** 2 for actual, target in zip((atom_count, ratio), target_values))

	param_space = {}
	for key in keys:
		split = key.split(":")
		min_value = int(split[1])
		max_value = int(split[2])
		param_space[key] = (min_value, max_value)
	optimizer = BayesianOptimization(
		f=result,
		pbounds=param_space,
		verbose=2,
		random_state=4,
		allow_duplicate_points=True
	)
	# Disable logging
	logging.getLogger("").setLevel(logging.ERROR)
	optimizer.maximize(init_points=5, n_iter=10)
	atom_count, ratio, nanoparticle = run_fuzzer(**optimizer.max['params'])
	logging.getLogger("").setLevel(config.LOG_LEVEL)
	rprint(optimizer.max['params'])
	rprint(f"Final result: {atom_count=}, {ratio=}")
	if plot:
		nanoparticle.plot()
