import os.path
import typer
from pathlib import Path

if os.path.exists(Path("config/config_local.py")):
    from config.config_local import *
    if 'MACHINES' not in globals():
        import setup_logging
        setup_logging.setup_logging()
        logging.fatal("Please add `from config_base import *` to config_local.py")
        raise typer.Exit(1)
else:
    from config.config_base import *
