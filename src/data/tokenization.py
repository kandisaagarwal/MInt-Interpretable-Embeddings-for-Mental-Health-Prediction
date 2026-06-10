import re


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text into lowercase alphabetic tokens."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"u\/\w+", "", text)
    text = re.sub(r"r\/\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [tok for tok in text.split() if tok]
