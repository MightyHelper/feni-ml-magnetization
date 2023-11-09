import datetime
import os
import multiprocessing


import rich.table
import rich.repr
import typer
from rich.highlighter import Highlighter, ReprHighlighter, RegexHighlighter
from rich.pretty import pprint

import poorly_coded_parser as parser
from rich import print as rprint

import executor

main = typer.Typer(add_completion=False, no_args_is_help=True)
shapefolder = typer.Typer(add_completion=False, no_args_is_help=True)
executions = typer.Typer(add_completion=False, no_args_is_help=True)
main.add_typer(shapefolder, name="shapefolder")
main.add_typer(executions, name="executions")
console = rich.console.Console()

h = ReprHighlighter()


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
	for i, key in enumerate(parser.recursive_input_search(path)):
		ptype, subtype, subsubtype = parse_nanoparticle_name(key)
		table.add_row(f"[green]{i}[/green]", f"[cyan]{key}[/cyan]", f"[blue]{ptype}[/blue]", f"[blue]{subtype}[/blue]", f"[blue]{subsubtype}[/blue]")
	console.print(table, highlight=True)


def parse_nanoparticle_name(key):
	parts = key.split("/")
	ptype = parts[2]
	subtype = parts[3]
	subsubtype = parts[4] if not parts[4].endswith(".in") else ""
	return ptype, subtype, subsubtype


@shapefolder.command()
def parseshapes(path: str = "../Shapes", threads: int = None, test: bool = True):
	"""
	Runs all nanoparticle simulations in a folder
	"""
	rprint(f"Parsing all input files in [bold underline green]{path}[/bold underline green]")
	if threads is None:
		if test:
			threads = multiprocessing.cpu_count()
		else:
			threads = multiprocessing.cpu_count() / 8
	nanoparticles = executor.execute_all_nanoparticles_in(path, threads, [], test)
	nanoparticles.drop(columns=["np"], inplace=True)
	table = rich.table.Table(title="Nanoparticle run results")
	for column in nanoparticles.columns:
		table.add_column(column)
	table.add_column("Type")
	table.add_column("SubType")
	table.add_column("SubSubType")
	for i in nanoparticles.index.values:
		ptype, subtype, subsubtype = parse_nanoparticle_name(nanoparticles.iloc[i]["key"])
		table.add_row(*[h(str(j)) for j in nanoparticles.iloc[i]], ptype, subtype, subsubtype)
	console.print(table, highlight=True)


@executions.command()
def list():
	"""
	List all executions that were done
	"""
	table = rich.table.Table(title="Executed simulations", show_footer=True)
	table.add_column("Index", justify="right", footer="Total")
	table.add_column("Folder Name", footer=str(len(os.listdir("../executions"))))
	table.add_column("Simulation Date")
	for i, folder in enumerate(os.listdir("../executions")):
		real_date = None
		if "_" in folder:
			sim, date = folder.split("_")
			# load date from unix timestamp
			real_date = datetime.datetime.utcfromtimestamp(float(date))
		table.add_row(f"[green]{i}[/green]", f"[blue]{folder}[/blue]", f"[bold]{real_date}[/bold]")
	console.print(table, highlight=True)


@executions.command()
def clean():
	"""
	Clean all executions
	"""
	total = 0
	for path in os.listdir("../executions"):
		for sub_path in os.listdir("../executions/" + path):
			os.remove("../executions/" + path + "/" + sub_path)
		os.rmdir("../executions/" + path)
		total += 1
	if total == 0:
		rprint(f"[red]No executions to remove[/red].")
	else:
		rprint(f"Removed [green]{total}[/green] executions.")


if __name__ == "__main__":
	main()
