from typing import List, Tuple, Set, Dict, Any
import re
from renard.pipeline.core import PipelineStep


class CustomSubstitutionPreprocessor(PipelineStep):
    """A preprocessor alowing regex-based substition

    :ivar substition_rules: A list of rules, each rule being of the
        form (match, substitution).
    """

    def __init__(self, substition_rules: List[Tuple[str, str]]) -> None:
        self.substition_rules = substition_rules

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        for rule in self.substition_rules:
            text = re.sub(rule[0], rule[1], text)
        return {"text": text}

    def needs(self) -> Set[str]:
        return set()

    def produces(self) -> Set[str]:
        return set()
