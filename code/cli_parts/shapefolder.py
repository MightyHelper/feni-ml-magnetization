import logging
from typing import Annotated
import pandas as pd
import rich.table
import typer
from rich import print as rprint

import config
import nanoparticle_locator
import poorly_coded_parser as parser
import service.executor_service
from cli_parts.number_highlighter import console, h
from nanoparticle import Nanoparticle
from service.executor_service import execute_nanoparticles
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


def lerp_green_red(value: float) -> str:
    # Return hex of color
    green = int(value * 255)
    red = int((1 - value) * 255)
    green_hex = hex(green)[2:]
    red_hex = hex(red)[2:]
    return f"#{red_hex:0>2}{green_hex:0>2}00".upper()


def correct_highlighter(column: str, value) -> str:
    desired_by_column = {
        "ratio_fe": config.DESIRED_FE_RATIO,
        "ratio_ni": config.DESIRED_NI_RATIO,
        "total": config.DESIRED_ATOM_COUNT,
        "fe": config.DESIRED_FE_ATOM_COUNT,
        "ni": config.DESIRED_NI_ATOM_COUNT
    }
    if column in desired_by_column:
        val = float(value)
        target = desired_by_column[column]
        acceptable_variance = 0.1 if target < 1 else 100
        x = 1 - min(abs(val - target) / acceptable_variance, 1)
        color = lerp_green_red(x)
        return f"[{color}]{value}[/{color}]"
    else:
        return h(value)


@shapefolder.command()
def parseshapes(
        path: str = "../Shapes",
        test: bool = True,
        seed_count: int = 1,
        seed: int = 123,
        count_only: bool = False,
        at: Annotated[str, "toko, toko:thread_count, local, local:thread_count"] = "local"
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

    df: pd.DataFrame = pd.DataFrame([nanoparticle.asdict() for _, nanoparticle in nanoparticles])
    df.drop(columns=["np"], inplace=True)
    table = rich.table.Table(title="Nanoparticle run results")
    for column in df.columns:
        table.add_column(column)
    table.add_column("Shape")
    table.add_column("Distribution")
    table.add_column("Interface")
    table.add_column("Pores")
    table.add_column("Index")
    for i in df.index.values:
        shape, distribution, interface, pores, index = parse_nanoparticle_name(df.iloc[i]["title"])
        table.add_row(*[correct_highlighter(table.columns[idx].header, str(j)) for idx, j in enumerate(df.iloc[i])],
                      shape, distribution, interface, pores, index)
    console.print(table, highlight=True)
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
