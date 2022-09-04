from typing import List
from dataclasses import dataclass


@dataclass
class CoreferenceMention:
    start_idx: int
    end_idx: int
    tokens: List[str]
