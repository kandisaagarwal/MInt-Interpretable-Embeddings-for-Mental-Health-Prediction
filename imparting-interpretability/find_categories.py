# Script to look up Roget category codes and names for a single input word.

import argparse

import roget

def main() -> None:
    """Look up Roget category codes and names for a single input word."""
    parser = argparse.ArgumentParser(
        description="Print Roget category codes and names for a given word."
    )
    parser.add_argument("--word", required=True)
    args = parser.parse_args()

    # Load Roget dictionaries (word->categories and code->name mappings).
    r = roget.roget.Roget()
    # Normalize case/whitespace before dictionary lookup.
    word = args.word.lower().strip()

    # Pull category codes for this word; empty list means no match in Roget.
    categories = r.thes_dict.get(word, [])
    if not categories:
        print(f"No categories found for: {args.word}")
        return

    # Print a compact summary, then each code with its human-readable label.
    print(f"Word: {args.word}")
    print(f"Found {len(categories)} categories")
    for code in categories:
        print(f"{code}\t{r.num_cat.get(code, '<unk>')}")


if __name__ == "__main__":
    main()
