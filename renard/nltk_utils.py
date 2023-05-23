from typing import List

#: ISO 639-3 language string correspondance with nltk language string
NLTK_ISO_STRING_TO_LANG = {
    "eng": "english",
    "ces": "czech",
    "dan": "danish",
    "nld": "dutch",
    "est": "estonian",
    "fin": "finnish",
    "fra": "french",
    "deu": "german",
    "ell": "greek",
    "ita": "italian",
    "nor": "norwegian",
    "pol": "polish",
    "por": "portuguese",
    "rus": "russian",
    "slv": "slovene",
    "spa": "spanish",
    "swe": "swedish",
    "tur": "turkish",
}

NLTK_LONG_TAG_TO_SHORT_TAG = {
    "B-PERSON": "B-PER",
    "I-PERSON": "I-PER",
    "B-ORGANIZATION": "B-ORG",
    "I-ORGANIZATION": "I-ORG",
    "B-LOCATION": "B-LOC",
    "I-LOCATION": "I-LOC",
    "O": "O",
}


def nltk_fix_bio_tags(nltk_tags: List[str]) -> List[str]:
    """Convert nltk BIO tags into their short version (e.g. B-PERSON => B-PER)"""
    return [
        NLTK_LONG_TAG_TO_SHORT_TAG.get(long_tag, long_tag) for long_tag in nltk_tags
    ]
