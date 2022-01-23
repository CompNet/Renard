from typing import List
from dataclasses import dataclass, field


@dataclass
class CoreferenceMention:
    start_idx: int
    end_idx: int
    mention: str


@dataclass
class CoreferenceChain:
    mentions: List[CoreferenceMention] = field(default_factory=lambda: [])
