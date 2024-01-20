import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import rich.table
import typer
from matplotlib import pyplot as plt
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
from cli_parts import ui_utils
from cli_parts.number_highlighter import console
from cli_parts.ui_utils import ZeroHighlighter, remove_old_tasks, add_new_tasks, update_tasks, \
    create_tasks
from nanoparticle import Nanoparticle
from nanoparticlebuilder import NanoparticleBuilder
from service.executor_service import execute_nanoparticles, add_extra_nanoparticles

executions = typer.Typer(add_completion=False, no_args_is_help=True)


@executions.command()
def ls(
        count: Annotated[bool, typer.Option(help="Whether to count the executions", show_default=True)] = False,
        plot_magnetism: Annotated[bool, typer.Option(help="Whether to plot the magnetism of the executions", show_default=True)] = False,
        save: Annotated[Path, typer.Option(help="Path to save the plot", show_default=True)] = None,
        by: Annotated[str, typer.Option(help="Parameter to sort the executions by", show_default=True)] = "Shape"
):
    """
    List available nanoparticle executions
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
        with (config.EXEC_LS_POOL_TYPE() as pool):
            result = pool.starmap(_load_single_nanoparticle, enumerate(sorted(execs)))
            for parts in result:
                if parts is None:
                    continue
                data, tab = parts
                df.loc[len(df)] = data  # Add new row to df using .loc
                table.add_row(*tab)
    df['magnetism_std_over_mag'] = df['magnetism_std'] / df['magnetism_val']
    console.print(table, highlight=True)
    if plot_magnetism:
        for by_value in by.split(","):
            logging.info(f"Plotting {by_value}")
            fig: plt.Figure = ui_utils.multi_plots(
                df,
                "Execution result",
                (by_value, 'magnetism_val', None, None),
                (by_value, "magnetism_std", None, None)
            )
            if save is not None:
                fig.savefig(os.path.join(save.as_posix(), f"exec_result_{by_value}.png"))
            else:
                plt.show()
            cli_parts.ui_utils.scatter(df, by=by_value, y="magnetism_std_over_mag", x="magnetism_val")
            plt.show()
            if save is not None:
                fig.savefig(os.path.join(save.as_posix(), f"exec_result_scatter_{by_value}.png"))
            else:
                plt.show()


def _load_single_nanoparticle(i: int, folder: str) -> tuple[dict[str, str], tuple[str, str, str, str, str, str]]:
    try:
        info = nanoparticle.Nanoparticle.from_executed(config.LOCAL_EXECUTION_PATH / folder)
        row = (
            f"[green]{i}[/green]",
            f"[cyan]{folder}[/cyan]",
            f"[blue]{info.title}[/blue]",
            f"[yellow]{datetime.utcfromtimestamp(float(info.get_simulation_date()))}[/yellow]",
            f"[magenta]{info.magnetism}[/magenta]",
            f"[red]{info.extra_replacements['in_toko']}[/red]"
        )
        out = utils.assign_nanoparticle_name(info.title)
        data = {
            **out,
            "magnetism_val": info.magnetism[0],
            "magnetism_std": info.magnetism[1]
        }
        return data, row
    except Exception as e:
        logging.debug(f"Error parsing {folder}: {e}")


@executions.command()
def clean(
        keep_ok: Annotated[bool, typer.Option(help="Whether to keep OK executions", show_default=True)] = False,
        keep_full: Annotated[bool, typer.Option(help="Whether to keep full executions", show_default=True)] = True,
        keep_batch: Annotated[bool, typer.Option(help="Whether to keep batch executions", show_default=True)] = True
):
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
        os.rmdir(config.LOCAL_EXECUTION_PATH / execution)
        total += 1
    if total == 0:
        rprint(f"[red]No executions to remove[/red].")
    else:
        rprint(f"Removed [green]{total}[/green] executions.")


@executions.command()
def live(
        in_toko: Annotated[bool, typer.Option(help="Whether to listen for executions in Toko", show_default=True)] = False,
        listen_anyway: Annotated[bool, typer.Option(help="Whether to listen for executions even if there are none", show_default=True)] = False,
        only_running: Annotated[bool, typer.Option(help="Whether to only listen for running executions", show_default=True)] = True
):
    """
    Find live executions
    """
    with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            expand=True,
            speed_estimate_period=600,
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
def execute(
        paths: Annotated[list[Path], typer.Option(help="List of paths to nanoparticle files", show_default=True)],
        plot: Annotated[bool, typer.Option(help="Whether to plot the nanoparticle or not", show_default=True)] = False,
        test: Annotated[bool, typer.Option(help="Whether to run in test mode or not", show_default=True)] = True,
        at: Annotated[str, typer.Option(help="Where to execute the nanoparticle simulation", show_default=True)] = "local",
        seed: Annotated[int, typer.Option(help="Seed for the random number generator", show_default=True)] = 123,
        seed_count: Annotated[int, typer.Option(help="Number of extra nanoparticles to add", show_default=True)] = 0
) -> list[Path | None]:
    """
    Execute a nanoparticle simulation
    """
    result_paths: list[Path | None] = []
    builders: list[tuple[str, NanoparticleBuilder]] = list(parser.PoorlyCodedParser.load_shapes_from_paths(paths))
    nanoparticles: list[tuple[str, Nanoparticle]] = add_extra_nanoparticles(builders, seed, seed_count)
    results: list[tuple[str, nanoparticle.Nanoparticle]] = execute_nanoparticles(nanoparticles, at, test)
    for path, nano in results:
        rprint(nano.asdict())
        if plot:
            nano.plot()
        result_paths.append(nano.local_path)
    return result_paths

@executions.command()
def inspect(
        paths: Annotated[list[Path], typer.Option(help="List of paths to nanoparticle files", show_default=True)],
        plot: Annotated[bool, typer.Option(help="Whether to plot the nanoparticle or not", show_default=True)] = False,
        csv: Annotated[bool, typer.Option(help="Whether to export nanoparticle data to a CSV file", show_default=True)] = False,
        g_r: Annotated[bool, typer.Option(help="Whether to calculate the radial distribution function g(r)", show_default=True)] = False,
        pec: Annotated[bool, typer.Option(help="Whether to calculate the potential energy curve", show_default=True)] = False,
        coord: Annotated[bool, typer.Option(help="Whether to calculate the coordination number", show_default=True)] = False,
        np_data: Annotated[bool, typer.Option(help="Whether to display nanoparticle data", show_default=True)] = False
):
    """
    Inspect a complete nanoparticle simulation
    """
    for path in paths:
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
def csv(
        paths: Annotated[list[Path], typer.Option(help="List of paths to nanoparticle files", show_default=True)],
        output_csv_format: Annotated[Path, typer.Option(help="Path to the output CSV format file", show_default=True)],
        concat: Annotated[bool, typer.Option(help="Whether to concatenate the output with existing CSV file", show_default=True)] = False
):
    """
    Export finished nanoparticle execution data to a CSV file
    """
    my_csv = pd.read_csv(output_csv_format)
    dfs = []
    for path in paths:
        nano = nanoparticle.Nanoparticle.from_executed(path)
        dfs.append(nano.columns_for_dataset())
    my_df = pd.concat(dfs)
    my_df = my_df[my_csv.columns]  # Sort my_df columns to be in the order of my_csv
    if concat:
        my_df = pd.concat([my_csv, my_df])  # Concat my_df and my_csv
    print(my_df.to_csv(index=False))  # raw print without row index


@executions.command()
def raw_parse_completed(reparse: Annotated[bool, typer.Option(help="Whether to reparse completed nanoparticle simulations", show_default=True)] = False):
    """
    Parse all completed nanoparticle simulations
    """
    with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            expand=True,
            refresh_per_second=20
    ) as progress:
        to_parse = []
        to_analyse = os.listdir(config.LOCAL_EXECUTION_PATH)
        gather_task = progress.add_task("Gathering", total=len(to_analyse))
        for i, folder in enumerate(to_analyse):
            if not os.path.exists(
                    os.path.join(config.LOCAL_EXECUTION_PATH, folder, f"iron.{config.FULL_RUN_DURATION}.dump")):
                continue
            if not reparse and os.path.exists(os.path.join(config.LOCAL_EXECUTION_PATH, folder, "magnetism.txt")):
                continue
            to_parse.append(folder)
            progress.update(gather_task, completed=i, total=len(to_parse))
        progress.remove_task(gather_task)
        task_id = progress.add_task("Parsing", total=len(to_parse))
        for i, folder in enumerate(to_parse):
            logging.info(f"Parsing {folder}")
            nano = nanoparticle.Nanoparticle.from_executed(config.LOCAL_EXECUTION_PATH / folder)
            nano.on_post_execution("Some non-empty result")
            progress.update(task_id, completed=i, total=len(to_parse))
        progress.remove_task(task_id)
