import datetime
import multiprocessing
import os
import re
import time
import platform
import rich.repr
import rich.table
import typer
import executor
import poorly_coded_parser as parser
import template
import nanoparticle
import logging
from rich import print as rprint
from rich.highlighter import ReprHighlighter
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.logging import RichHandler

logging.basicConfig(
	level="NOTSET",
	format="%(message)s",
	datefmt="[%X]",
	handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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


def parse_execution_info(folder):
	out = {
		'real_date': None,
		'title': None,
		'mag': None
	}
	if "_" in folder:
		parts = folder.split("_")
		sim, date = parts[0], parts[1]
		out['real_date'] = datetime.datetime.utcfromtimestamp(float(date))
	out['title'] = get_execution_title(folder)
	out['mag'] = get_magnetism(folder)
	return out


def get_execution_title(folder):
	try:
		with open("../executions/" + folder + "/nanoparticle.in", "r") as f:
			lines = f.readlines()
			return lines[0][2:].strip()
	except FileNotFoundError:
		pass
	return "Unknown"


def get_magnetism(folder):
	try:
		with open("../executions/" + folder + "/magnetism.txt", "r") as f:
			lines = f.readlines()
			return lines[1].strip()
	except FileNotFoundError:
		pass
	return "Unknown"


@executions.command()
def ls():
	"""
	List all executions that were done
	"""
	table = rich.table.Table(title="Executed simulations", show_footer=True)
	table.add_column("Index", justify="right", footer="Total")
	table.add_column("Folder Name", footer=str(len(os.listdir("../executions"))))
	table.add_column("Simulation Date")
	table.add_column("Title")
	table.add_column("Magnetism")

	for i, folder in enumerate(sorted(os.listdir("../executions"))):
		info = parse_execution_info(folder)
		table.add_row(
			f"[green]{i}[/green]",
			f"[blue]{folder}[/blue]",
			f"[bold]{info['real_date']}[/bold]",
			f"[magenta]{info['title']}[/magenta]",
			f"[bold green]{info['mag']}[/bold green]" if info['mag'] != "Unknown" else f"[bold red]{info['mag']}[/bold red]"
		)
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


def get_current_step(lammps_log):
	"""
	Get the current step of a lammps log file
	"""
	step = -1
	try:
		with open(lammps_log, "r") as f:
			lines = f.readlines()
			try:
				split = re.split(r" +", lines[-1].strip())
				step = int(split[0])
			except Exception:
				pass
	except FileNotFoundError:
		pass
	return step


def get_title(path):
	title = "Unknown"
	try:
		with open(path, "r") as f:
			lines = f.readlines()
			title = lines[0][2:].strip()
	except FileNotFoundError:
		pass
	return title


def get_runninng_windows():
	# wmic.exe process where "name='python.exe'" get commandline
	result = os.popen("wmic.exe process where \"name='python.exe'\" get commandline").readlines()
	print(result)
	for execution in {x for result in result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
		parts = execution.split("/")
		foldername = '/'.join(parts[:-1])
		step = get_current_step(foldername + "/log.lammps")
		title = get_title(foldername + "/nanoparticle.in")
		yield foldername, step, title


def get_running_executions():
	if platform.system() == "Windows":
		yield from get_runninng_windows()
	elif platform.system() == "Linux":
		yield from get_runninng_linux()
	else:
		raise Exception(f"Unknown system: {platform.system()}")


def get_runninng_linux():
	ps_result = os.popen("ps -ef | grep " + template.LAMMPS_EXECUTABLE).readlines()
	for execution in {x for result in ps_result if (x := re.sub(".*?(-in (.*))?\n", "\\2", result)) != ""}:
		parts = execution.split("/")
		foldername = '/'.join(parts[:-1])
		step = get_current_step(foldername + "/log.lammps")
		title = get_title(foldername + "/nanoparticle.in")
		yield foldername, step, title


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
			logging.info("[yellow]Exiting...[/yellow]")


@executions.command()
def execute(path: str, plot: bool = False, test: bool = True):
	"""
	Execute a nanoparticle simulation
	"""
	_, nano = parser.parse_single_shape(path)
	nano = nano.build()
	nano.execute(test_run=test)
	rprint(executor.parse_ok_execution_results(path, nano, test))
	if plot:
		nano.plot()


def add_task(folder, progress: Progress, step, tasks, title):
	logging.info(f"Found running execution: {folder} ({step})")
	tasks[folder] = progress.add_task(f"{folder} ({title})", total=None if step == -1 else nanoparticle.FULL_RUN_DURATION)


if __name__ == "__main__":
	main()
