import logging
import multiprocessing
from typing import Annotated

import pandas as pd
import rich.table
import typer
from rich import print as rprint

import execution_queue
import executor
import nanoparticle_locator
import poorly_coded_parser as parser
import toko_utils
from cli_parts.number_highlighter import console, h
from nanoparticle import Nanoparticle
from utils import parse_nanoparticle_name, dot_dot

shapefolder = typer.Typer(add_completion=False, no_args_is_help=True, name="shapefolder")


@shapefolder.command()
def ls(path: str = "../Shapes"):
	"""
	List available nanoparticles in folder
	"""
	table = rich.table.Table(title="Available nanoparticles")
	table.add_column("Index")
	table.add_column("Path")
	table.add_column("Type")
	table.add_column("SubType")
	table.add_column("SubSubType")
	table.add_column("R")
	for i, (path, nano) in enumerate(parser.PoorlyCodedParser.load_shapes(path, [])):
		ptype, subtype, subsubtype = parse_nanoparticle_name(path)
		table.add_row(
			f"[green]{i}[/green]",
			f"[cyan]{path}[/cyan]",
			f"[blue]{ptype}[/blue]",
			f"[blue]{subtype}[/blue]",
			f"[blue]{subsubtype}[/blue]",
			f"[green]{len(nano.seed_values)}[/green]" if nano.is_random() else "[red]0[/red]"
		)
	console.print(table, highlight=True)


def _run(x: execution_queue.ExecutionQueue):
	x.run()

@shapefolder.command()
def parseshapes(path: str = "../Shapes", threads: int = None, test: bool = True, seed_count: int = 1, seed: int = 123, count_only: bool = False, at: Annotated[str, "toko, toko:queue_name, local"] = "local"):
	"""
	Runs all nanoparticle simulations in a folder
	"""
	rprint(f"Parsing all input files in [bold underline green]{path}[/bold underline green]")

	nanoparticles: list[tuple[str, Nanoparticle]] = executor.build_nanoparticles_to_execute([], path, seed, seed_count)
	if count_only:
		rprint(f"Found [green]{len(nanoparticles)}[/green] nanoparticle shapes.")
		return
	queues = [get_executor(at) for _ in range(threads)]
	for i, (key, np) in enumerate(nanoparticles):
		np.schedule_execution(execution_queue=queues[i % threads], test_run=test)
	# Run each queue in a separate thread
	threads = [multiprocessing.Process(target=_run, args=(q,)) for q in queues]

	for t in threads:
		t.start()

	for t in threads:
		t.join()

	df: pd.DataFrame = pd.DataFrame([nanoparticle.asdict() for _, nanoparticle in nanoparticles])
	df.drop(columns=["np"], inplace=True)
	table = rich.table.Table(title="Nanoparticle run results")
	for column in df.columns:
		table.add_column(column)
	table.add_column("Type")
	table.add_column("SubType")
	table.add_column("SubSubType")
	for i in df.index.values:
		ptype, subtype, subsubtype = parse_nanoparticle_name(df.iloc[i]["key"])
		table.add_row(*[h(str(j)) for j in df.iloc[i]], ptype, subtype, subsubtype)
	console.print(table, highlight=True)


def schedule_with_queue(nanoparticle, at, test):
	queue = get_executor(at)
	key, np = nanoparticle
	np.schedule_execution(execution_queue=queue, test_run=test)
	queue.run()
	return nanoparticle


def get_executor(at):
	if at == "local":
		queue = execution_queue.LocalExecutionQueue()
	elif at == "toko":
		queue = toko_utils.TokoExecutionQueue()
	elif at.startswith("toko:"):
		queue = toko_utils.TokoBatchedExecutionQueue(int(at.split(":")[1]))
	else:
		raise ValueError(f"Unknown queue {at}")
	return queue


@shapefolder.command()
def inspect(path: str):
	"""
	Inspect a nanoparticle
	"""
	_, nano = parser.PoorlyCodedParser.parse_single_shape(path)
	is_random = nano.is_random()
	nano = nano.build()
	region = nano.get_region()
	rprint(f"[bold underline green]Can seeds be modified?[/bold underline green] {is_random}")
	rprint(nano)
	rprint(region)


@shapefolder.command()
def shrink():
	"""
	Shrink all nanoparticle shapes
	"""
	for path in nanoparticle_locator.NanoparticleLocator.sorted_search("../Shapes"):
		_, nano = parser.PoorlyCodedParser.parse_single_shape(path)
		nano = nano.build()
		region = nano.get_region()
		shrink_path = dot_dot(path) + "/nano.shrink"
		with open(shrink_path, "w") as f:
			f.write(region)
		_, parsed = parser.PoorlyCodedParser.parse_single_shape(shrink_path, True)
		parsed = parsed.build()
		parsed_region = parsed.get_region()
		assert parsed_region == region, f"Regions are not equal:\n{parsed_region}\n{region}"
		logging.info(f"Shrunk {path}")
