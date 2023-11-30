import logging
import multiprocessing
import os
import platform
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import rich.repr
import rich.table
import typer
from rich import print as rprint
from rich.columns import Columns
from rich.console import Group
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.theme import Theme

import config
import executor
import nanoparticle
import poorly_coded_parser as parser
from config import LOG_LEVEL
from utils import get_running_executions, ZeroHighlighter, parse_nanoparticle_name, add_task, dot_dot

# Don't turn these signal into exceptions, just die.
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

logging.basicConfig(
	level="NOTSET",
	format="%(message)s",
	datefmt="[%X]",
	handlers=[RichHandler(rich_tracebacks=True)]
)

if not sys.stdout.isatty():
	logging.disable(logging.DEBUG)  # Disable debug and info messages
log = logging.getLogger("rich")
logging.getLogger("").setLevel(LOG_LEVEL)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

main = typer.Typer(add_completion=False, no_args_is_help=True)
shapefolder = typer.Typer(add_completion=False, no_args_is_help=True, name="shapefolder")
executions = typer.Typer(add_completion=False, no_args_is_help=True)
main.add_typer(shapefolder, name="sf")
main.add_typer(executions, name="exec")
col = "#333333"
console = rich.console.Console(theme=Theme({
	"zero.zero": col,
	"zero.zero_1": col,
	"zero.zero_2": col,
	"zero.zero_3": col,
}))

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
	table.add_column("Random")
	for i, (path, nano) in enumerate(parser.load_shapes(path, [])):
		ptype, subtype, subsubtype = parse_nanoparticle_name(path)
		table.add_row(f"[green]{i}[/green]", f"[cyan]{path}[/cyan]", f"[blue]{ptype}[/blue]", f"[blue]{subtype}[/blue]", f"[blue]{subsubtype}[/blue]", f"[green]True[/green]" if nano.is_random() else "[red]False[/red]")
	console.print(table, highlight=True)


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


@shapefolder.command()
def inspect(path: str):
	"""
	Inspect a nanoparticle
	"""
	_, nano = parser.parse_single_shape(path)
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
	for path in parser.sorted_recursive_input_search("../Shapes"):
		_, nano = parser.parse_single_shape(path)
		nano = nano.build()
		region = nano.get_region()
		shrink_path = dot_dot(path) + "/nano.shrink"
		with open(shrink_path, "w") as f:
			f.write(region)
		_, parsed = parser.parse_single_shape(shrink_path, True)
		parsed = parsed.build()
		parsed_region = parsed.get_region()
		assert parsed_region == region, f"Regions are not equal:\n{parsed_region}\n{region}"
		logging.info(f"Shrunk {path}")


@executions.command()
def ls():
	"""
	List all executions that were done
	"""
	table = rich.table.Table(title="Executed simulations", show_footer=True)
	table.add_column("Index", justify="right", footer="Total")
	execs = os.listdir(config.LOCAL_EXECUTION_PATH)
	table.add_column("Folder Name", footer=str(len(execs)))
	table.add_column("Title")
	table.add_column("Date")
	table.add_column("Magnetism")
	table.add_column("In Toko")

	for i, folder in enumerate(sorted(execs)):

		info = nanoparticle.Nanoparticle.from_executed(config.LOCAL_EXECUTION_PATH + "/" + folder)
		table.add_row(
			f"[green]{i}[/green]",
			f"[cyan]{folder}[/cyan]",
			f"[blue]{info.title}[/blue]",
			f"[yellow]{datetime.utcfromtimestamp(float(info.get_simulation_date()))}[/yellow]",
			f"[magenta]{info.magnetism}[/magenta]",
			f"[red]{info.extra_replacements['in_toko']}[/red]"
		)
	console.print(table, highlight=True)


@executions.command()
def clean():
	"""
	Clean all executions
	"""
	total = 0
	for path in os.listdir(config.LOCAL_EXECUTION_PATH):
		for sub_path in os.listdir(config.LOCAL_EXECUTION_PATH + "/" + path):
			os.remove(config.LOCAL_EXECUTION_PATH + "/" + path + "/" + sub_path)
		os.rmdir(config.LOCAL_EXECUTION_PATH + "/" + path)
		total += 1
	if total == 0:
		rprint(f"[red]No executions to remove[/red].")
	else:
		rprint(f"Removed [green]{total}[/green] executions.")


@executions.command()
def live():
	"""
		Find live executions
	"""
	# Run ps -ef | grep lmp
	with Progress(
		SpinnerColumn(),
		*Progress.get_default_columns(),
		MofNCompleteColumn(),
		TimeElapsedColumn(),
		expand=True
	) as progress:
		running = [*get_running_executions()]
		tasks = {}
		for folder, step, title in running:
			add_task(folder, progress, step, tasks, title)
		if len(running) == 0:
			logging.error("[red]No running executions found[/red]", extra={"markup": True})
			return
		try:
			while True:
				for folder, step, title in running:
					progress.update(tasks[folder], completed=step, total=None if step == -1 else nanoparticle.FULL_RUN_DURATION)
				progress.refresh()
				time.sleep(0.2)
				running = [*get_running_executions()]
				for folder, step, title in running:
					if folder not in tasks:
						add_task(folder, progress, step, tasks, title)
				keys_to_remove = []
				for folder in tasks.keys():
					if folder not in [x for x, _, _ in running]:
						logging.info(f"Execution {folder} ({step}) has finished")
						try:
							progress.remove_task(tasks[folder])
							keys_to_remove.append(folder)
						except KeyError:
							pass
				for key in keys_to_remove:
					del tasks[key]
		except KeyboardInterrupt:
			logging.info("[yellow]Exiting...[/yellow]", extra={"markup": True})


@executions.command()
def execute(path: str, plot: bool = False, test: bool = True, in_toko: bool = False):
	"""
	Execute a nanoparticle simulation
	"""
	_, nano = parser.parse_single_shape(path)
	nano = nano.build()
	nano.execute(test_run=test, in_toko=in_toko)
	rprint(executor.parse_ok_execution_results(path, nano, test))
	if plot:
		nano.plot()


@executions.command()
def inspect(
	paths: list[Path],
	plot: bool = False,
	csv: bool = False,
	g_r: bool = False,
	pec: bool = False,
	coord: bool = False,
	np_data: bool = False
):
	"""
	Inspect a complete nanoparticle simulation
	"""
	for path in paths:
		if platform.system() == "Linux":
			path = path.absolute().as_posix()
		else:
			path = path.absolute().as_uri()
		nano = nanoparticle.Nanoparticle.from_executed(path)

		reh = ZeroHighlighter()
		r = ReprHighlighter()
		if g_r or pec or coord or np_data:
			console.print(Panel.fit(Group(
				Columns([x for x in [
					Panel.fit(reh(nano.psd.to_string()), title="Total g(r)", border_style="green") if g_r else "",
					Panel.fit(reh(nano.psd_p.to_string()), title="Partial g(r)", border_style="green") if g_r else "",
					Panel.fit(reh(nano.pec.to_string()), title="Potential energy", border_style="blue") if pec else "",
				] if x != ""], expand=False),
				Columns([
					Panel.fit(reh(nano.get_full_coord().to_string()), title="Coordination number", border_style="cyan") if coord else "",
					Panel.fit(
						Group(
							r(f"magnetism={nano.magnetism}"),
							r(f"total={nano.total}"),
							r(f"fe_s={nano.fe_s}"),
							r(f"ni_s={nano.ni_s}"),
							r(f"fe_c={nano.fe_c}"),
							r(f"ni_c={nano.ni_c}")
						), title="Nanoparticle data") if np_data else ""
				], expand=False)
			), title=nano.title))
		if csv:
			data = nano.columns_for_dataset()
			print(data.to_csv(index=False))
		if plot:
			nano.plot()


@executions.command()
def csv(paths: list[Path], output_csv_format: Path):
	my_csv = pd.read_csv(output_csv_format)
	dfs = []
	for path in paths:
		if platform.system() == "Linux":
			path = path.absolute().as_posix()
		else:
			path = path.absolute().as_uri()
		nano = nanoparticle.Nanoparticle.from_executed(path)
		dfs.append(nano.columns_for_dataset())
	my_df = pd.concat(dfs)
	# Sort my_df columns to be in the order of my_csv
	my_df = my_df[my_csv.columns]
	# Concat my_df and my_csv
	my_df = pd.concat([my_csv, my_df])
	# print without row index
	print(my_df.to_csv(index=False))


if __name__ == "__main__":
	main()
