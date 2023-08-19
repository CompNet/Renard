male_titles = {
    "eng": {"mr.", "mr", "mister", "m.", "m", "sir", "lord"},
    "fra": {"monsieur", "mr", "mr.", "seigneur", "duc", "comte", "m", "m.", "sire"},
}

female_titles = {
    "eng": {"miss", "mrs.", "mrs", "lady"},
    "fra": {"madame", "mademoiselle", "dame", "mme", "mlle"},
}

all_titles = {
    key: male_titles[key].union(female_titles[key]) for key in male_titles.keys()
}


def is_a_male_title(title: str, lang: str = "eng") -> bool:
    try:
        return title.lower() in male_titles[lang]
    except KeyError:
        raise ValueError(
            f"unsupported lang for is_a_male_title: {lang} (supported langs: {list(male_titles.keys())})"
        )


def is_a_female_title(title: str, lang: str = "eng") -> bool:
    try:
        return title.lower() in female_titles[lang]
    except KeyError:
        raise ValueError(
            f"unsupported lang for is_a_female_title: {lang} (supported langs: {list(female_titles.keys())})"
        )
