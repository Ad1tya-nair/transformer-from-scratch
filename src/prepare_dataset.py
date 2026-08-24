from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "tokenizer.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tokenized"
)


# ============================================================
# CONFIGURATION
# ============================================================

SPLITS = [
    "train",
    "validation",
    "test"
]


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    tokenizer,
    split_name
):

    input_file = (
        SPLIT_DIR
        / f"{split_name}.txt"
    )

    output_file = (
        OUTPUT_DIR
        / f"{split_name}.bin"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing input file:\n{input_file}"
        )

    print(f"\nProcessing {split_name}...")

    # --------------------------------------------------------
    # Read text
    # --------------------------------------------------------

    text = input_file.read_text(
        encoding="utf-8"
    )

    print(
        f"Characters: {len(text):,}"
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    encoded = tokenizer.encode(text)

    token_ids = encoded.ids

    print(
        f"Tokens: {len(token_ids):,}"
    )

    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    token_array = np.array(
        token_ids,
        dtype=np.uint16
    )

    # --------------------------------------------------------
    # Save binary file
    # --------------------------------------------------------

    token_array.tofile(
        output_file
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        f"File size: "
        f"{output_file.stat().st_size / (1024 * 1024):.2f} MB"
    )

    return len(token_ids)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not TOKENIZER_FILE.exists():

        raise FileNotFoundError(
            f"Tokenizer not found:\n"
            f"{TOKENIZER_FILE}"
        )

    print("=" * 60)
    print("DATASET TOKENIZATION")
    print("=" * 60)

    print(
        f"Tokenizer:\n"
        f"{TOKENIZER_FILE}"
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print(
        f"\nVocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    total_tokens = {}

    for split_name in SPLITS:

        token_count = process_split(
            tokenizer,
            split_name
        )

        total_tokens[split_name] = token_count

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)

    for split_name in SPLITS:

        print(
            f"{split_name:<12}"
            f"{total_tokens[split_name]:>12,} tokens"
        )


if __name__ == "__main__":
    main()