# Script to create a custom Roget word group based on either a seed word or a category, and map it to vocab index.

import argparse
from pathlib import Path
from typing import Optional
import json

import roget


def load_vocab(vocab_file: Path):
    """Read vocab.txt and map each word to its 1-based line index.

    The training code expects word IDs to be 1-indexed, so we keep that here.
    """
    vocab = [line.split(' ')[0].lower() for line in vocab_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    vocab_to_id = {word: idx + 1 for idx, word in enumerate(vocab)}
    return vocab_to_id


def build_group_by_seed_word(r, seed_word: str):
    """Build a group by taking all Roget categories attached to a seed word.

    Example: if seed word is "happy", collect words from every category
    that contains "happy".
    """
    seed = seed_word.lower()
    if seed not in r.thes_dict:
        raise ValueError(f"Seed word not found in Roget thesaurus: {seed_word}")
    cat_codes = r.thes_dict[seed]
    words = []
    for cat in cat_codes:
        words.extend(r.thes_cat.get(cat, []))
    group_name = f"SEED:{seed_word}"
    return group_name, words


def build_group_by_category(r, category: str):
    """Build a group directly from a Roget category code or category name."""
    cat = category.strip()
    cat_code = None

    if cat in r.thes_cat:
        cat_code = cat
    else:
        lookup = cat.upper()
        if lookup in r.cat_num:
            cat_code = r.cat_num[lookup]

    if cat_code is None:
        raise ValueError(f"Category not found. Use a code like cat0647 or a category name.")

    group_name = r.num_cat.get(cat_code, cat_code)
    words = r.thes_cat.get(cat_code, [])
    return group_name, words


def create_single_word_group(
    vocab_file: Path,
    seed_word: Optional[str] = None,
    category: Optional[str] = None,
):
    """Create a Roget word group, map to vocab IDs, and write output files.

    Returns a dict with group metadata and output file paths.
    """
    if not seed_word and not category:
        raise ValueError("Provide either seed_word or category")

    r = roget.roget.Roget()

    vocab_to_id = load_vocab(vocab_file)

    if seed_word:
        # print(f"Building group from seed word: {seed_word}")
        group_name, group_words = build_group_by_seed_word(r, seed_word)
        source = "seed_word"
        input_value = seed_word
    else:
        #print(f"Building group from category: {category}")
        group_name, group_words = build_group_by_category(r, category)
        source = "category"
        input_value = category

    # Keep only words that actually exist in the chosen vocabulary
    # This avoids forcing IDs that the model cannot see during training.
    cleaned = []
    for word in group_words:
        low = word.lower()
        if low in vocab_to_id:
            cleaned.append((low, vocab_to_id[low]))

    # Deduplicate words and keep the best (lowest) vocab ID for each word.
    # Lower IDs generally correspond to more frequent words in vocab.txt
    best_id_by_word = {}
    for word, wid in cleaned:
        if word not in best_id_by_word or wid < best_id_by_word[word]:
            best_id_by_word[word] = wid

    # Sort by vocab ID so output stays deterministic and frequency-friendly
    pairs = sorted(best_id_by_word.items(), key=lambda x: x[1])
    word_ids = [wid for _, wid in pairs]


    words = [w for w, _ in pairs]
    ids = [str(wid) for _, wid in pairs]

    return words, ids, group_name

def create_groups(
    vocab_file: Path,
    concepts: list[str],
):
    """Create multiple Roget word groups based on a list of concepts."""
    results = []
    for concept in concepts:
        # print(f"\nCreating group for concept: {concept}")
        try:
            words, ids, group_name = create_single_word_group(
                vocab_file=vocab_file,
                seed_word=concept,
                category=None,
            )
            results.append((words, ids, group_name))
        except Exception as e:
            print(f"Error creating group for concept '{concept}': {e}")
    out_dir = Path(f"word_groups_out_COMBINED")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_file = out_dir / f"ids.txt"
    words_file = out_dir / f"words_readable.json"

    # Create files for to write to
    ids_file.write_text("", encoding="utf-8")
    words_file.write_text("", encoding="utf-8")

    # Write ids.txt
    ids_lines = [" ".join(ids) for _, ids, _ in results]
    ids_file.write_text("\n".join(ids_lines) + ("\n" if ids_lines else ""), encoding="utf-8")

    # Write human-readable JSON 
    json_records = [
        {"name": group_name, "length": len(words), "words": words}
        for words, _, group_name in results
    ]
    words_file.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")

    return results

def main():
    # This script supports two paths:
    # 1) --seed_word (derive categories from a word), or
    # 2) --category (pick a category explicitly)
    # 3) --concepts (create multiple groups from a list of seed words)
    parser = argparse.ArgumentParser(description="Create one custom Roget word group and map words to vocab IDs.")
    parser.add_argument("--vocab_file", type=Path, required=True, help="Path to vocab file (word count format)")
    parser.add_argument("--seed_word", type=str, default=None, help="Seed word to derive a group from all its Roget categories")
    parser.add_argument("--category", type=str, default=None, help="Roget category code/name to use directly")
    parser.add_argument("--concepts_file", nargs="+", help="List of seed words to create groups from")
    args = parser.parse_args()

    if args.concepts_file:
        print("Creating multiple groups from concepts file...")
        concepts = []
        for file in args.concepts_file:
            with open(file, "r") as f:
                concepts.extend([line.strip() for line in f if line.strip()])
        create_groups(
            vocab_file=args.vocab_file,
            concepts=concepts,
        )
    else:
        words, ids, group_name = create_single_word_group(
            vocab_file=args.vocab_file,
            seed_word=args.seed_word,
            category=args.category,
        )
        # Save single group outputs
        out_dir = Path(f"word_groups_out_{group_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        ids_file = out_dir / f"{group_name}_ids.txt"
        words_file = out_dir / f"{group_name}_words.txt"
        ids_file.write_text(" ".join(str(x) for x in ids) + "\n", encoding="utf-8")
        words_file.write_text("\n".join(words) + ("\n" if words else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
