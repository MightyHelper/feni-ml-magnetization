import logging
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from matplotlib import pyplot as plt
from rich import print as rprint

import config
from cli_parts import ui_utils
from nanoparticle_renamer import NanoparticleRenamer, BasicNanoparticleRenamer, NewNanoparticleRenamer
from utils import assign_nanoparticle_name

dat = typer.Typer(add_completion=False, no_args_is_help=True, name="dataset")


@dat.command()
def rename(path: Annotated[Path, typer.Option(help="Path to the folder", show_default=True)], rename_type: Annotated[str, typer.Option(help="Type of renaming to be done", show_default=True)] = "basic"):
    """
    Outputs the renamed nanoparticles for a given folder.
    """
    class_type = _get_renamer(rename_type)

    renames: list[tuple[str, str]] = class_type.get_all_renames_for_folder(path)
    if len(renames) == 0:
        rprint("[yellow]No renames found[/yellow]")
    class_type.output_renames(renames)


def _get_renamer(rename_type):
    """
    Returns the appropriate renamer class based on the rename_type.
    """
    class_type = NanoparticleRenamer
    if rename_type == "basic":
        class_type = BasicNanoparticleRenamer
    elif rename_type == "new":
        class_type = NewNanoparticleRenamer
    return class_type


@dat.command()
def single(path: Annotated[Path, typer.Option(help="Path to the folder", show_default=True)], rename_type: Annotated[str, typer.Option(help="Type of renaming to be done", show_default=True)] = "basic"):
    """
    Outputs the renamed nanoparticle for a single folder.
    """
    class_type = _get_renamer(rename_type)
    nanoparticle = class_type.get_new_nanoparticle_name(path.as_posix(), [])
    rprint(f"[green]{nanoparticle}[/green]")


@dat.command()
def rename_in_dataset(dataset_path: Annotated[Path, typer.Option(help="Path to the dataset", show_default=True)], output_path: Annotated[Path, typer.Option(help="Path to output the renamed dataset", show_default=True)] = None, rename_type: Annotated[str, typer.Option(help="Type of renaming to be done", show_default=True)] = "basic"):
    """
    Outputs the renamed nanoparticles for a dataset.
    """
    class_type = _get_renamer(rename_type)

    dataset: pd.DataFrame = pd.read_csv(dataset_path)
    names = dataset['name'].tolist()
    renames: list[tuple[str, str]] = class_type.get_all_renames(names)
    if len(renames) == 0:
        rprint("[yellow]No renames found[/yellow]")
    for old_name, new_name in renames:
        dataset.loc[dataset['name'] == old_name, 'name'] = new_name
        rprint(f"[blue]{old_name}[/blue] -> [green]{new_name}[/green]")
    if output_path is not None:
        dataset.to_csv(output_path, index=False)


@dat.command()
def normalize_ratios(input_path: Annotated[Path, typer.Option(help="Path to the input dataset", show_default=True)], output_path: Annotated[Path, typer.Option(help="Path to output the normalized dataset", show_default=True)] = None):
    """
    Normalizes the ratios in the dataset and outputs the result.
    """
    dataset: pd.DataFrame = pd.read_csv(input_path)
    rprint("[magenta]Before[/magenta]")
    rprint(dataset[['fe_s', 'ni_s', 'fe_c', 'ni_c', 'n_fe', 'n_ni']].to_string())
    # Find rows where n_fe and n_ni > 1, and then normalise with total = n_fe + n_ni
    dataset['n_fe'] = dataset['n_fe'].astype(float)
    dataset['n_ni'] = dataset['n_ni'].astype(float)
    dataset['n_total'] = dataset['n_fe'] + dataset['n_ni']
    dataset['n_fe'] = dataset['n_fe'] / dataset['n_total']
    dataset['n_ni'] = dataset['n_ni'] / dataset['n_total']
    rprint("[green]After[/green]")
    rprint(dataset[['fe_s', 'ni_s', 'fe_c', 'ni_c', 'n_fe', 'n_ni']].to_string())
    if output_path is not None:
        dataset.to_csv(output_path, index=False)


@dat.command()
def dataset_info(
        dataset_path: Annotated[Path, typer.Argument(help="Path to dataset")],
        by: Annotated[str, typer.Option(help="Csv of Fields to group by", show_default=True)] = "Shape",
        save: Annotated[Path, typer.Option(help="Path to save the plot", show_default=True)] = None
):
    """
    Outputs information about the dataset grouped by the specified fields.
    """
    dataset = pd.read_csv(dataset_path)
    for i, row in dataset.iterrows():
        result = assign_nanoparticle_name(row['name'])
        for key, value in result.items():
            dataset.loc[i, key] = value
    for by_value in by.split(","):
        logging.info(f"Plotting {by_value}")
        fig: plt.Figure = ui_utils.multi_plots(
            dataset,
            dataset_path.name,
            (by_value, 'tmg', None, None),
            (by_value, "n_ni", config.DESIRED_NI_RATIO - config.DESIRED_MAX_RATIO_VARIANCE,
             config.DESIRED_NI_RATIO + config.DESIRED_MAX_RATIO_VARIANCE)
        )
        if save is not None:
            fig.savefig(os.path.join(save.as_posix(), f"{dataset_path.name}_{by_value}.png"))
        else:
            plt.show()
