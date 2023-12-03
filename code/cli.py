import logging
import signal
import sys

import typer
from rich.logging import RichHandler

from cli_parts.executions import executions
from cli_parts.shapefolder import shapefolder
from cli_parts.fuzzer import fuzzer
from config import LOG_LEVEL

# Don't turn these signal into exceptions, just die.
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

logging.basicConfig(
	level="NOTSET",
	format="%(message)s",
	datefmt="[%X]",
	handlers=[RichHandler(rich_tracebacks=True)]
)

if not sys.stdout.isatty():
	logging.disable(logging.DEBUG)  # Disable debug and info messages
log = logging.getLogger("rich")
logging.getLogger("").setLevel(LOG_LEVEL)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

main = typer.Typer(add_completion=False, no_args_is_help=True)

main.add_typer(shapefolder, name="sf")
main.add_typer(executions, name="exec")
main.add_typer(fuzzer, name="fuz")

if __name__ == "__main__":
	main()
