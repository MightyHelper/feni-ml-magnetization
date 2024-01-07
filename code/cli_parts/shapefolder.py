import logging
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import rich.table
import typer
from rich import print as rprint

import config
import nanoparticle_locator
import poorly_coded_parser as parser
import service.executor_service
from cli_parts.number_highlighter import console
from nanoparticle import Nanoparticle
from service.executor_service import execute_nanoparticles
from utils import parse_nanoparticle_name, dot_dot
from cli_parts.ui_utils import do_plots, correct_highlighter

shapefolder = typer.Typer(add_completion=False, no_args_is_help=True, name="shapefolder")


@shapefolder.command()
def ls(path: str = "../Shapes"):
    """
    List available nanoparticles in folder
    """
    table = rich.table.Table(title="Available nanoparticles")
    table.add_column("Index")
    table.add_column("Path")
    table.add_column("Shape")
    table.add_column("Distribution")
    table.add_column("Interface")
    table.add_column("Pores")
    table.add_column("Index")
    table.add_column("R")
    for i, (path, nano) in enumerate(parser.PoorlyCodedParser.load_shapes(path, [])):
        shape, distribution, interface, pores, index = parse_nanoparticle_name(path)
        table.add_row(
            f"[green]{i}[/green]",
            f"[cyan]{path}[/cyan]",
            f"[blue]{shape}[/blue]",
            f"[blue]{distribution}[/blue]",
            f"[blue]{interface}[/blue]",
            f"[blue]{pores}[/blue]",
            f"[blue]{index}[/blue]",
            f"[green]{len(nano.seed_values)}[/green]" if nano.is_random() else "[red]0[/red]"
        )
    console.print(table, highlight=True)


@shapefolder.command()
def parseshapes(
        path: Annotated[
            str,
            typer.Option(
                help="Path to folder with nanoparticle shapes",
                show_default=True
            )
        ] = "../Shapes",
        test: Annotated[
            bool,
            typer.Option(
                help=f"Test run (run 0 instead of run {config.FULL_RUN_DURATION})",
                show_default=True
            )
        ] = True,
        seed_count: Annotated[
            int,
            typer.Option(
                help="Number of nanoparticles to generate with different seeds per existing nanoparticle",
                show_default=True
            )
        ] = 1,
        seed: Annotated[
            int,
            typer.Option(
                help="Base seed for other seeds",
                show_default=True
            )
        ] = 123,
        count_only: Annotated[
            bool,
            typer.Option(
                help="Only count the number of nanoparticle shapes without executing",
                show_default=True
            )
        ] = False,
        at: Annotated[
            str,
            typer.Option(
                help="Possible values: [b u]toko[/b u], [b u]toko:thread_count[/b u], [b u]local[/b u], [b u]local:thread_count[/b u]",
                show_default=True
            )
        ] = "local",
        plot_ni_distribution: Annotated[
            bool,
            typer.Option(
                help="Plot the distribution of Ni ratio across nanoparticles",
                show_default=True
            )
        ] = False,
        full_column_names: Annotated[
            bool,
            typer.Option(
                help="Use full column names",
                show_default=True
            )
        ] = False,
        sort_by: Annotated[
            str,
            typer.Option(
                help="Sort by [b u]title[/b u] or [b u]errors[/b u]",
                show_default=True
            )
        ] = "title",
        show_simulation_key: Annotated[
            bool,
            typer.Option(
                help="Show the simulation key (folder name)",
                show_default=True
            )
        ] = False,
) -> list[tuple[str, Nanoparticle]] | int:
    """
    Runs all nanoparticle simulations in a folder
    """
    rprint(f"Parsing all input files in [bold underline green]{path}[/bold underline green]")

    nanoparticles: list[tuple[str, Nanoparticle]] = service.executor_service.build_nanoparticles_to_execute(
        [],
        path,
        seed,
        seed_count
    )
    if count_only:
        rprint(f"Found [green]{len(nanoparticles)}[/green] nanoparticle shapes.")
        return len(nanoparticles)
    nanoparticles = execute_nanoparticles(nanoparticles, at, test)
    table = rich.table.Table(title="Nanoparticle run results", show_footer=True)
    df: pd.DataFrame = pd.DataFrame([nanoparticle.asdict() for _, nanoparticle in nanoparticles])
    df.drop(columns=["np"], inplace=True)
    if not show_simulation_key:
        df.drop(columns=["key"], inplace=True)
    count_ok_col = "Count OK" if full_column_names else 'C'
    ratio_ok_col = "Ratio OK" if full_column_names else 'R'
    df[count_ok_col] = (abs(df["total"] - config.DESIRED_ATOM_COUNT) < config.DESIRED_MAX_ATOM_COUNT_VARIANCE)
    df[ratio_ok_col] = (abs(df["ratio_fe"] - config.DESIRED_FE_RATIO) < config.DESIRED_MAX_RATIO_VARIANCE)
    if sort_by == "title":
        df = df.sort_values(by=["title"], ignore_index=True)
    elif sort_by == "errors":
        df = df.sort_values(by=[ratio_ok_col, count_ok_col, "title"], ignore_index=True, ascending=[False, False, True])
    # if all df['mag'] == (None, None) dont show the column
    if all(df['mag'] == (None, None)):
        df.drop(columns=["mag"], inplace=True)
    for column in df.columns:
        full_names = {
            "ratio_fe": "Fe Ratio",
            "ratio_ni": "Ni Ratio",
            "total": "Total atoms",
            "fe": "Fe atoms",
            "ni": "Ni atoms",
            "mag": "Magnetism",
            "title": "Title",
            "ok": "OK",
            "key": "Simulation folder",
        }
        short_names = {
            "ratio_fe": "rFe",
            "ratio_ni": "rNi",
            "total": "T",
            "fe": "Fe",
            "ni": "Ni",
            "mag": "Mag",
            "title": "Title",
            "ok": "OK",
            "key": "Folder",
        }
        if column in full_names:
            column = (full_names if full_column_names else short_names)[column]
        if column in [ratio_ok_col, count_ok_col]:
            value = len(df) - df[column].sum()
            if value == 0:
                table.add_column(column, footer=f"[green]{value}[/green]")
            else:
                table.add_column(column, footer=f"[red]{value}[/red]")
        else:
            table.add_column(column)
    table.add_column("Shape")
    table.add_column("Distribution")
    table.add_column("Int")
    table.add_column("Pores")
    table.add_column("Index" if full_column_names else "I")
    shapes = []
    distributions = []
    interfaces = []
    pores_v = []
    indexes = []
    for i in df.index.values:
        shape, distribution, interface, pores, index = parse_nanoparticle_name(df.iloc[i]["title"])
        shapes.append(shape)
        distributions.append(distribution)
        interfaces.append(interface)
        pores_v.append(pores)
        indexes.append(index)
        table.add_row(
            *[
                correct_highlighter(df.columns[idx], str(j))
                for idx, j in enumerate(df.iloc[i])
            ],
            shape,
            distribution,
            interface,
            pores,
            index
        )
    df["Shape"] = shapes
    df["Distribution"] = distributions
    df["Interface"] = interfaces
    df["Pores"] = pores_v
    df["Index"] = indexes
    console.print(table, highlight=True)
    if plot_ni_distribution:
        # Plot ratio_ni
        fig: plt.Figure = do_plots(
            df,
            "Shape",
            "ratio_ni",
            config.DESIRED_NI_RATIO - config.DESIRED_MAX_RATIO_VARIANCE,
            config.DESIRED_NI_RATIO + config.DESIRED_MAX_RATIO_VARIANCE
        )
        plt.show()

    return nanoparticles


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


@shapefolder.command(deprecated=True, hidden=True)
def shrink():
    """
    Shrink all nanoparticle shapes (Deprecated - Do not use
    )
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
