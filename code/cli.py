import typer
import setup_logging
from cli_parts.executions import executions
from cli_parts.shapefolder import shapefolder
from cli_parts.fuzzer import fuzzer

setup_logging.setup_logging()

main = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")

main.add_typer(shapefolder, name="sf", help="Commands for nanoparticle shape folder")
main.add_typer(executions, name="exec", help="Commands for nanoparticle executions")
main.add_typer(fuzzer, name="fuz", help="Commands for nanoparticle fuzzing")

if __name__ == "__main__":
	main()
