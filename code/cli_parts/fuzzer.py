import logging
import multiprocessing

import rich.table
import typer
from rich import print as rprint

import executor
import poorly_coded_parser as parser
from cli_parts.number_highlighter import console, h
from utils import parse_nanoparticle_name, dot_dot

fuzzer = typer.Typer(add_completion=False, no_args_is_help=True, name="fuzzer")
