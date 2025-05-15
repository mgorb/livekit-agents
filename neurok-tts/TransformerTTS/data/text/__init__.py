from typing import Union

from data.text.symbols import all_phonemes
from data.text.tokenizer import Phonemizer, Tokenizer


class TextToTokens:
    def __init__(self, phonemizer: Phonemizer, tokenizer: Tokenizer):
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer

    def __call__(self, input_text: Union[str, list], speaker_id: Union[int, str]) -> list:
        phons = self.phonemizer(input_text)
        tokens = self.tokenizer(phons, speaker_id)
        return tokens

    @classmethod
    def default(cls, language: str, add_start_end: bool, with_stress: bool, model_breathing: bool, njobs=1,
                alphabet=None, collapse_whitespace: bool = True, gst: bool = False, zfill: int = 0):
        phonemizer = Phonemizer(language=language, njobs=njobs, with_stress=with_stress, alphabet=alphabet,
                                collapse_whitespace=collapse_whitespace)
        tokenizer = Tokenizer(add_start_end=add_start_end, model_breathing=model_breathing, alphabet=alphabet,
                              gst=gst, zfill=zfill)
        return cls(phonemizer=phonemizer, tokenizer=tokenizer)
