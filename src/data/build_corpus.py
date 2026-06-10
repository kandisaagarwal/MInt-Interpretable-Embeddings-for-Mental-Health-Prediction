import argparse
import json
from pathlib import Path

from tokenization import tokenize
from tqdm import tqdm



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "corpus.txt"


def build_corpus(input_dir: Path, output_file: Path, input_glob: str = "*.jsonl", text_key: str = "text") -> dict[str, int]:
    """Combine raw files into a tokenized one-line-per-document corpus file."""
    files = sorted(input_dir.glob(input_glob))
    if not files:
        raise ValueError(f"No files matched pattern '{input_glob}' in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    docs_written = 0
    docs_seen = 0
    total_tokens = 0

    # Use total line count so the progress bar reflects full file traversal.
    total_lines = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            total_lines += sum(1 for _ in f)

    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_lines, desc="Building corpus", unit="line")

    with output_file.open("w", encoding="utf-8") as out:
        for file_path in files:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if progress is not None:
                        progress.update(1)

                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    docs_seen += 1
                    text = obj.get(text_key, "")
                    if not text or text in {"[deleted]", "[removed]"}:
                        continue

                    tokens = tokenize(text)
                    if not tokens:
                        continue

                    out.write(" ".join(tokens) + "\n")
                    docs_written += 1
                    total_tokens += len(tokens)

    if progress is not None:
        progress.close()

    return {
        "files": len(files),
        "docs_seen": docs_seen,
        "docs_written": docs_written,
        "tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a combined tokenized corpus from raw JSONL files.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing raw JSONL files")
    parser.add_argument("--input_glob", type=str, default="*.jsonl", help="Glob pattern for raw files")
    parser.add_argument("--text_key", type=str, default="text", help="JSON key that stores document text")
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Output combined corpus file")
    args = parser.parse_args()

    stats = build_corpus(
        input_dir=args.input_dir,
        output_file=args.output_file,
        input_glob=args.input_glob,
        text_key=args.text_key,
    )

    print("Corpus written:", args.output_file)
    print("Summary:")
    print(f"  Files processed: {stats['files']}")
    print(f"  JSON records read: {stats['docs_seen']}")
    print(f"  Documents written: {stats['docs_written']}")
    print(f"  Tokens written: {stats['tokens']}")


if __name__ == "__main__":
    main()
