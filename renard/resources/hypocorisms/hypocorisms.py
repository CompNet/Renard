from typing import Dict, List, Set, Tuple
from collections import defaultdict
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))


class HypocorismGazetteer:
    """An hypocorism (nicknames) gazetteer

    .. note::

        datas used for this gazeeter come from
        https://github.com/carltonnorthern/nickname-and-diminutive-names-lookup
        and are licensed under the Apache 2.0 License
    """

    supported_langs = {"eng"}

    def __init__(self, lang: str = "eng"):
        """
        :param lang: gazetteer language.  Must be in
            ``HypocorismGazetteer.supported_langs``.
        """
        if not lang in HypocorismGazetteer.supported_langs:
            print(
                f"[warning] {lang} not supported by {type(self)} (supported languages: {HypocorismGazetteer.supported_langs})",
                file=sys.stderr,
            )

        self.name_to_nicknames = defaultdict(set)
        self.nickname_to_names = defaultdict(set)

        if lang == "eng":
            with open(f"{script_dir}/datas/hypocorisms.csv") as f:
                for line in f:
                    # it should be illegal to parse csv like that,
                    # however in this specific case we know there
                    # are no issues... right ?
                    line = line.strip()
                    splitted = line.split(",")
                    name = splitted[0]
                    nicknames = splitted[1:]

                    self._add_hypocorism_(name, nicknames)

    def _add_hypocorism_(self, name: str, nicknames: List[str]):
        """Add a name associated with several nicknames

        :param name:
        :param nicknames: nicknames to associate to the given name
        """
        name = name.lower()
        nicknames = [n.lower() for n in nicknames]
        for nickname in nicknames:
            self.nickname_to_names[nickname].add(name)
            self.name_to_nicknames[name].add(nickname)

    def get_nicknames(self, name: str) -> Set[str]:
        """Return all possible nickname for the given name"""
        try:
            return self.name_to_nicknames[name.lower()]
        except KeyError:
            return set()

    def get_possible_names(self, nickname: str) -> Set[str]:
        """Return all names that can correspond to the given nickname"""
        try:
            return self.nickname_to_names[nickname.lower()]
        except KeyError:
            return set()

    def are_related(self, name1: str, name2: str) -> bool:
        """Check if one name is an hypocorism of the other
        (or if both names are equals)
        """
        if name1 == "" or name2 == "":
            return False

        return (
            name1.lower() == name2.lower()
            or name2.lower() in self.get_nicknames(name1)
            or name2.lower() in self.get_possible_names(name1)
        )
