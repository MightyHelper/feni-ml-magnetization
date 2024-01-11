# Don't turn these signal into exceptions, just die.
import logging
import signal
import sys

from rich.logging import RichHandler


def setup_logging():
	global LOG_LEVEL
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
	if 'LOG_LEVEL' in globals():
		logging.getLogger("").setLevel(LOG_LEVEL)
	else:
		logging.getLogger("").setLevel(logging.WARNING)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	return logging.getLogger("rich")
