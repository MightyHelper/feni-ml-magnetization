import typer
from pathlib import Path
from rich import print as rprint
from nanoparticle_renamer import NanoparticleRenamer

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
