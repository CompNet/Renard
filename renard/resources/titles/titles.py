import re

male_titles = {
    "eng": [r"[Mm]r\.?", r"[Mm]\.?", r"[Ss]ir", r"[Ll]ord"],
    "fra": [r"[Mm]onsieur", r"[Mm]r\.?", r"[Ss]eigneur"],
}

female_titles = {
    "eng": [r"[Mm]iss", r"[Mm]r?s\.?", r"[Ll]ady"],
    "fra": [r"[Mm]adame", r"[Mm]ademoiselle", "[Dd]ame"],
}


def is_a_male_title(title: str, lang: str = "eng") -> bool:
    try:
        pattern_list = male_titles[lang]
        return any([re.match(pattern, title) for pattern in pattern_list])
    except KeyError:
        raise ValueError(
            f"unsupported lang for is_a_male_title: {lang} (supported langs: {list(male_titles.keys())})"
        )


def is_a_female_title(title: str, lang: str = "eng") -> bool:
    try:
        pattern_list = female_titles[lang]
        return any([re.match(pattern, title) for pattern in pattern_list])
    except KeyError:
        raise ValueError(
            f"unsupported lang for is_a_female_title: {lang} (supported langs: {list(female_titles.keys())})"
        )
