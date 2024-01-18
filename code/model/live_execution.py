from dataclasses import dataclass
from pathlib import PurePath


@dataclass
class LiveExecution:
    title: str
    step: int
    folder: PurePath
