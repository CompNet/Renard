import sys
import renard.pipeline.character_unification as cu

print(
    "[warning] the characters_extraction module is deprecated. Use character_unification instead.",
    file=sys.stderr,
)

Character = cu.Character
GraphRulesCharactersExtractor = cu.GraphRulesCharacterUnifier
NaiveCharactersExtractor = cu.NaiveCharacterUnifier
_assign_coreference_mentions = cu._assign_coreference_mentions
