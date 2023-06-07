import re

males_pronouns = {
    "eng": {"he", "him", "his", "himself"},
    "fra": {"il", "son", "lui-même"},
}
females_pronouns = {
    "eng": {"she", "her", "hers", "herself"},
    "fra": {"elle", "sa", "elle-même"},
}


def is_a_male_pronoun(word: str, lang: str = "eng") -> bool:
    try:
        return word.lower() in males_pronouns[lang]
    except KeyError:
        raise ValueError(
            f"unsupported lang for is_a_male_pronoun: {lang} (supported langs: {list(males_pronouns.keys())})"
        )


def is_a_female_pronoun(word: str, lang: str = "eng") -> bool:
    try:
        return word.lower() in females_pronouns[lang]
    except KeyError:
        raise ValueError(
            f"unsupported lang for is_a_female_pronoun: {lang} (supported langs: {list(females_pronouns.keys())})"
        )
