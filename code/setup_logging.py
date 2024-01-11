# Don't turn these signal into exceptions, just die.
import logging
import signal
import sys

from rich.logging import RichHandler
from config import LOG_LEVEL


def setup_logging():
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	if hasattr(signal, "SIGPIPE"):
		signal.signal(signal.SIGPIPE, signal.SIG_DFL)

	logging.basicConfig(
		level="NOTSET",
		format="%(message)s",
		datefmt="[%X]",
		handlers=[RichHandler(rich_tracebacks=True)]
	)

	if not sys.stdout.isatty():
		logging.disable(logging.DEBUG)  # Disable debug and info messages
	logging.getLogger("").setLevel(LOG_LEVEL)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	return logging.getLogger("rich")
