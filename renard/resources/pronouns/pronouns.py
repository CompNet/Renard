males_pronouns = {"he", "him", "his", "himself"}
females_pronouns = {"she", "her", "hers", "herself"}


def is_a_male_pronoun(word: str) -> bool:
    return word.lower() in males_pronouns


def is_a_female_pronoun(word: str) -> bool:
    return word.lower() in females_pronouns
