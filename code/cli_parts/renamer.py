import pandas as pd
import typer
from pathlib import Path
from rich import print as rprint

import config
from cli_parts import ui_utils
from nanoparticle_renamer import NanoparticleRenamer
from utils import parse_nanoparticle_name

renamer = typer.Typer(add_completion=False, no_args_is_help=True, name="renamer")


@renamer.command()
def rename(path: Path = Path("../Shapes")):
    """
    Output nanoparticle renames for a folder
    """
    renames: list[tuple[str, str]] = NanoparticleRenamer.get_all_renames_for_folder(path.as_posix())
    if len(renames) == 0:
        rprint("[yellow]No renames found[/yellow]")
    NanoparticleRenamer.output_renames(renames)

@renamer.command()
def single(path: Path):
    """
    Output nanoparticle renames for a folder
    """
    nanoparticle = NanoparticleRenamer.get_new_nanoparticle_name(path.as_posix(), [])
    rprint(f"[green]{nanoparticle}[/green]")

@renamer.command()
def rename_in_dataset(dataset_path: Path, output_path: Path | None = None):
    """
    Output nanoparticle renames for a folder
    """
    dataset = pd.read_csv(dataset_path)
    names = dataset['name'].tolist()
    renames: list[tuple[str, str]] = NanoparticleRenamer.get_all_renames(names)
    if len(renames) == 0:
        rprint("[yellow]No renames found[/yellow]")
    for old_name, new_name in renames:
        dataset.loc[dataset['name'] == old_name, 'name'] = new_name
        rprint(f"[blue]{old_name}[/blue] -> [green]{new_name}[/green]")
    if output_path is not None:
        dataset.to_csv(output_path, index=False)

@renamer.command()
def normalize_ratios(input_path: Path, output_path: Path | None = None):
    """
    Output nanoparticle renames for a folder
    """
    dataset = pd.read_csv(input_path)
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

@renamer.command()
def dataset_info(dataset_path: Path):
    """
    Output nanoparticle renames for a folder
    """
    dataset = pd.read_csv(dataset_path)
    for i, row in dataset.iterrows():
        shape, distribution, interface, pores, index = parse_nanoparticle_name(row['name'])
        dataset.loc[i, 'Shape'] = shape
        dataset.loc[i, 'Distribution'] = distribution
        dataset.loc[i, 'Interface'] = interface
        dataset.loc[i, 'Pores'] = pores
        dataset.loc[i, 'Index'] = index
    ui_utils.do_plots(
        dataset,
        "Shape",
        "n_ni",
        config.DESIRED_NI_RATIO - config.DESIRED_MAX_RATIO_VARIANCE,
        config.DESIRED_NI_RATIO + config.DESIRED_MAX_RATIO_VARIANCE
    )

