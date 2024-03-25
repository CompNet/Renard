from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Set, List, Dict, Any, Union, Literal
import torch
from renard.pipeline.core import PipelineStep
from renard.pipeline.progress import ProgressReporter
from renard.pipeline.character_unification import Character
from renard.pipeline.quote_detection import Quote
from grimbert.model import SpeakerAttributionModel
from grimbert.predict import predict_speaker
from grimbert.datas import (
    SpeakerAttributionDataset,
    SpeakerAttributionDocument,
    SpeakerAttributionQuote,
    SpeakerAttributionMention,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast


class BertSpeakerDetector(PipelineStep):
    """Detect quote speaker in text"""

    QUOTE_CTX_LEN = 512
    SPEAKER_REPR_NB = 4

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, str]] = None,
        batch_size: int = 4,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ):
        if isinstance(model, str):
            self.huggingface_model_id = model
            self.model = None  # model will be init by _pipeline_init_
        else:
            self.huggingface_model_id = None
            self.model = model

        self.tokenizer = tokenizer

        self.batch_size = batch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        super().__init__()

    def _pipeline_init_(self, lang: str, **kwargs):
        from transformers import AutoTokenizer

        super()._pipeline_init_(lang, **kwargs)

        if self.model is None:
            # the user supplied a huggingface ID: load model from the HUB
            if not self.huggingface_model_id is None:
                self.model = SpeakerAttributionModel.from_pretrained(
                    self.huggingface_model_id
                )
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.huggingface_model_id
                    )
                self.lang = "unknown"

            # the user did not supply anything: load the default model
            else:
                self.model = SpeakerAttributionModel.from_pretrained(
                    "compnet-renard/spanbert-base-cased-literary-speaker-attribution"
                )
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        assert not self.tokenizer is None

    def __call__(
        self,
        tokens: List[str],
        quotes: List[Quote],
        characters: List[Character],
        **kwargs,
    ) -> Dict[str, Any]:
        id_to_char = {str(id(char)): char for char in characters}

        doc = SpeakerAttributionDocument(
            tokens,
            [
                SpeakerAttributionQuote(quote.tokens, quote.start, quote.end, None)
                for quote in quotes
            ],
            [
                SpeakerAttributionMention(
                    mention.tokens,
                    mention.start_idx,
                    mention.end_idx,
                    str(id(character)),
                )
                for character in characters
                for mention in character.mentions
            ],
        )
        dataset = SpeakerAttributionDataset(
            [doc],
            BertSpeakerDetector.QUOTE_CTX_LEN,
            BertSpeakerDetector.SPEAKER_REPR_NB,
            self.tokenizer,
        )

        raw_preds = predict_speaker(
            dataset,
            self.model,
            self.tokenizer,
            self.batch_size,
            device=self.device,
            quiet=True,
        )[0]
        preds = []
        for pred in raw_preds:
            if pred.score > 0.5:
                preds.append(id_to_char.get(pred.predicted_speaker))
            else:
                preds.append(None)

        assert len(preds) == len(quotes)

        return {"speakers": preds}

    def needs(self) -> Set[str]:
        """quotes, tokens, characters"""
        return {"tokens", "quotes", "characters"}

    def production(self) -> Set[str]:
        """speaker"""
        return {"speakers"}
