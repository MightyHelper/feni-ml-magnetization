import re
from dataclasses import dataclass
from pathlib import PurePath

from config import config


@dataclass
class LiveExecution:
    title: str
    step: int
    folder: PurePath

    def is_running(self):
        return self.step != -1

    def get_total_execution_length(self):
        if config.BATCH_EXECUTION in self.title:
            # Parse `BATCH_EXECUTION (total)` using regex
            pattern = r"\((\d+)\)"
            match = re.search(pattern, self.title)
            return None if match is None else int(match.group(1)) * config.FULL_RUN_DURATION

        return None if self.step == -1 else config.FULL_RUN_DURATION

    def __str__(self):
        return f"[{self.title} - {self.step}]"


