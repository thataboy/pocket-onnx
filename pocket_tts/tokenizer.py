import sentencepiece as spm
import re

class Tokenizer:
    def __init__(self, path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

    def encode(self, text: str) -> list[int]:
        text = text.strip()
        if not text:
            return []
        if text[-1].isalnum():
            text += "."
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return self.sp.encode_as_ids(text)

def split_sentences(text: str) -> list[str]:
    """Splits text into sentences based on punctuation."""
    # Matches . ! ? followed by whitespace or end of string, avoiding common abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'”’]|$)', text)
    return [s.strip() for s in sentences if s.strip()]