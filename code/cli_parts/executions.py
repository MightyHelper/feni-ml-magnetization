import logging
import os
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
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn, TaskID

import cli_parts.ui_utils
import config
import nanoparticle
import poorly_coded_parser as parser
import utils
from service.executor_service import execute_nanoparticles
from utils import resolve_path
from cli_parts.ui_utils import ZeroHighlighter, remove_old_tasks, add_new_tasks, update_tasks, \
    create_tasks
from cli_parts.number_highlighter import console

executions = typer.Typer(add_completion=False, no_args_is_help=True)


@executions.command()
def ls(count: bool = False, plot_magnetism: bool = False):
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
    df = pd.DataFrame(columns=["Shape", "magnetism_val", "magnetism_std"])
    if not count:
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
                shape, distribution, interface, pores, index = utils.parse_nanoparticle_name(info.title)
                df = df._append({
                    "Shape": shape,
                    "magnetism_val": info.magnetism[0],
                    "magnetism_std": info.magnetism[1]
                }, ignore_index=True)
            except Exception as e:
                logging.debug(f"Error parsing {folder}: {e}")
    if plot_magnetism:
        cli_parts.ui_utils.do_plots(df, by="Shape", field="magnetism_val")
        cli_parts.ui_utils.do_plots(df, by="Shape", field="magnetism_std")
    console.print(table, highlight=True)


@executions.command()
def clean(keep_ok: bool = False, keep_full: bool = True, keep_batch: bool = True):
    """
    Clean all executions
    """
    total = 0
    for execution in os.listdir(config.LOCAL_EXECUTION_PATH):
        if 'batch' in execution and keep_batch:
            continue
        listdir = os.listdir(os.path.join(config.LOCAL_EXECUTION_PATH, execution))
        if 'iron.0.dump' in listdir and keep_ok:
            continue
        if f'iron.{config.FULL_RUN_DURATION}.dump' in listdir and keep_full:
            continue
        for file in listdir:
            full_path = os.path.join(config.LOCAL_EXECUTION_PATH, execution, file)
            os.remove(full_path)
        os.rmdir(config.LOCAL_EXECUTION_PATH + "/" + execution)
        total += 1
    if total == 0:
        rprint(f"[red]No executions to remove[/red].")
    else:
        rprint(f"Removed [green]{total}[/green] executions.")


@executions.command()
def live(in_toko: bool = False, listen_anyway: bool = False, only_running: bool = True):
    """
    Find live executions
    :param in_toko: Whether to listen for executions in Toko
    :param listen_anyway: Whether to listen for executions even if there are none
    """
    with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            expand=True
    ) as progress:
        running: list[tuple[str, int, str]] = get_running_executions(in_toko, only_running)
        tasks: dict[str, TaskID] = {}
        create_tasks(progress, running, tasks)
        if len(running) == 0:
            logging.error("[red]No running executions found[/red]", extra={"markup": True})
            if not listen_anyway: return
        try:
            while True:
                update_tasks(progress, running, tasks)
                sleep_time: float = 5 + 0.25 * len(running) if in_toko else 0.2
                if in_toko: logging.debug(f"Waiting {sleep_time}")
                time.sleep(sleep_time)
                running = get_running_executions(in_toko, only_running)
                add_new_tasks(progress, running, tasks)
                remove_old_tasks(progress, running, tasks)
        except KeyboardInterrupt:
            logging.info("[yellow]Exiting...[/yellow]", extra={"markup": True})


def get_running_executions(in_toko: bool, only_running: bool) -> list[tuple[str, int, str]]:
    return [
        (folder, step, title)
        for folder, step, title in
        nanoparticle.RunningExecutionLocator.get_running_executions(in_toko)
        if step != -1 or not only_running
    ]


@executions.command()
def execute(path: Path, plot: bool = False, test: bool = True, at: str = "local") -> str | None:
    """
    Execute a nanoparticle simulation
    """
    abs_path: str = resolve_path(path)
    path, nano_builder = parser.PoorlyCodedParser.parse_single_shape(abs_path)
    nano: nanoparticle.Nanoparticle = nano_builder.build()
    result: list[tuple[str, nanoparticle.Nanoparticle]] = execute_nanoparticles([(path, nano)], at, test)
    try:
        nano = result[0][1]
        rprint(nano.asdict())
        if plot:
            nano.plot()
        return nano.path
    except IndexError:
        logging.error("Execution failed.")
        return None


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
        nano = nanoparticle.Nanoparticle.from_executed(resolve_path(path))

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
                    Panel.fit(reh(nano.get_full_coord().to_string()), title="Coordination number",
                              border_style="cyan") if coord else "",
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
        else:
            rprint(nano.asdict())
        if plot:
            nano.plot()


@executions.command()
def csv(paths: list[Path], output_csv_format: Path):
    my_csv = pd.read_csv(output_csv_format)
    dfs = []
    for path in paths:
        nano = nanoparticle.Nanoparticle.from_executed(resolve_path(path))
        dfs.append(nano.columns_for_dataset())
    my_df = pd.concat(dfs)
    my_df = my_df[my_csv.columns]  # Sort my_df columns to be in the order of my_csv
    my_df = pd.concat([my_csv, my_df])  # Concat my_df and my_csv
    print(my_df.to_csv(index=False))  # raw print without row index
