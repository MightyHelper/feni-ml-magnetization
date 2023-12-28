import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import rich.table
import typer
from rich import print as rprint
from rich.columns import Columns
from rich.console import Group
from rich.highlighter import ReprHighlighter
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn

import config
import executor
import nanoparticle
import poorly_coded_parser as parser
from service.executor_service import execute_nanoparticles
from utils import add_task, ZeroHighlighter, resolve_path
from cli_parts.number_highlighter import console

executions = typer.Typer(add_completion=False, no_args_is_help=True)


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
		try:
			info = nanoparticle.Nanoparticle.from_executed(config.LOCAL_EXECUTION_PATH + "/" + folder)
			table.add_row(
				f"[green]{i}[/green]",
				f"[cyan]{folder}[/cyan]",
				f"[blue]{info.title}[/blue]",
				f"[yellow]{datetime.utcfromtimestamp(float(info.get_simulation_date()))}[/yellow]",
				f"[magenta]{info.magnetism}[/magenta]",
				f"[red]{info.extra_replacements['in_toko']}[/red]"
			)
		except Exception as e:
			logging.debug(f"Error parsing {folder}: {e}")
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
def live(in_toko: bool = False, listen_anyway: bool = False):
	"""
	Find live executions
	:param in_toko: Whether to listen for executions in Toko
	:param listen_anyway: Whether to listen for executions even if there are none
	"""
	# Run ps -ef | grep lmp
	with Progress(
		SpinnerColumn(),
		*Progress.get_default_columns(),
		MofNCompleteColumn(),
		TimeElapsedColumn(),
		expand=True
	) as progress:
		running = [*nanoparticle.RunningExecutionLocator.get_running_executions(in_toko)]
		tasks = {}
		for folder, step, title in running:
			add_task(folder, progress, step, tasks, title)
		if len(running) == 0:
			logging.error("[red]No running executions found[/red]", extra={"markup": True})
			if not listen_anyway:
				return
		try:
			while True:
				for folder, step, title in running:
					progress.update(tasks[folder], completed=step, total=None if step == -1 else config.FULL_RUN_DURATION)
				progress.refresh()
				time.sleep(10 + 1 * len(running) if in_toko else 0.2)
				running = [*nanoparticle.RunningExecutionLocator.get_running_executions(in_toko)]
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
def execute(path: Path, plot: bool = False, test: bool = True, at: str = "local") -> None:
	"""
	Execute a nanoparticle simulation
	"""
	abs_path: str = resolve_path(path)
	path, nano_builder = parser.PoorlyCodedParser.parse_single_shape(abs_path)
	nano = nano_builder.build()
	result = execute_nanoparticles([(path, nano)], at, test)
	rprint(executor.parse_ok_execution_results(abs_path, nano, test))
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
