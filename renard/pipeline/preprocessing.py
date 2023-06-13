from typing import List, Tuple, Set, Dict, Any, Literal, Union
import re
from renard.pipeline.core import PipelineStep


class CustomSubstitutionPreprocessor(PipelineStep):
    """A preprocessor allowing regex-based substition"""

    def __init__(self, substition_rules: List[Tuple[str, str]]) -> None:
        """
        :param substition_rules: A list of rules, each rule being of the
            form (match, substitution).
        """
        self.substition_rules = substition_rules
        super().__init__()

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        :param text:
        """
        for rule in self.substition_rules:
            text = re.sub(rule[0], rule[1], text)
        return {"text": text}

    def needs(self) -> Set[str]:
        return set()

    def production(self) -> Set[str]:
        return set()

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return "any"
