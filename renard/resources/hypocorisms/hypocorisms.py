from typing import Dict, Set
from collections import defaultdict
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


class HypocorismGazetteer:
    """An hypocorism (nicknames) gazetteer

    .. note::

        datas used for this gazeeter come from
        https://github.com/carltonnorthern/nickname-and-diminutive-names-lookup
        and are licensed under the Apache 2.0 License
    """

    def __init__(self):

        self.name_to_nicknames = {}
        self.nickname_to_names = defaultdict(set)

        with open(f"{script_dir}/datas/hypocorisms.csv") as f:

            for line in f:

                # it should be illegal to parse csv like that,
                # however in this specific case we know there
                # are no issues... right ?
                line = line.strip()
                splitted = line.split(",")
                name = splitted[0]
                nicknames = splitted[1:]

                self.name_to_nicknames[name] = set(nicknames)
                for nickname in nicknames:
                    self.nickname_to_names[nickname].add(name)

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
        return (
            name1.lower() == name2.lower()
            or name2.lower() in self.get_nicknames(name1)
            or name2.lower() in self.get_possible_names(name1)
        )
