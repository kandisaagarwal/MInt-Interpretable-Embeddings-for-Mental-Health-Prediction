import argparse
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CORPUS_FILE = PROJECT_ROOT / "data" / "processed" / "corpus.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "vocab.txt"
DEFAULT_VOCAB_SIZE = 100000


def build_vocab(corpus_file: Path, output_file: Path, vocab_size: int = DEFAULT_VOCAB_SIZE) -> dict[str, int]:
    """Build vocab.txt from a tokenized corpus file."""
    counter = Counter()
    documents = 0
    total_tokens = 0

    with corpus_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = [tok for tok in line.strip().split() if tok]
            if not tokens:
                continue
            documents += 1
            total_tokens += len(tokens)
            counter.update(tokens)

    vocab = counter.most_common(vocab_size)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for word, count in vocab:
            f.write(f"{word},{count}\n")

    return {
        "documents": documents,
        "tokens": total_tokens,
        "unique_tokens": len(counter),
        "vocab_written": len(vocab),
    }


def main():
    parser = argparse.ArgumentParser(description="Build vocab from a tokenized corpus file.")
    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE, help="Path to tokenized corpus file")
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Output vocab file path")
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE, help="Maximum vocab size")
    args = parser.parse_args()

    stats = build_vocab(
        corpus_file=args.corpus_file,
        output_file=args.output_file,
        vocab_size=args.vocab_size,
    )

    print("Vocabulary saved to:", args.output_file)
    print("Corpus size summary:")
    print(f"  Documents: {stats['documents']}")
    print(f"  Tokens: {stats['tokens']}")
    print(f"  Unique tokens: {stats['unique_tokens']}")
    print(f"  Vocab entries written: {stats['vocab_written']}")


if __name__ == "__main__":
    main()
