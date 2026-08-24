from pathlib import Path
import json
import random


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"


# ============================================================
# WORD COUNT
# ============================================================

def count_words(file_path):
    """
    Count whitespace-separated words in a processed book.
    """

    text = file_path.read_text(
        encoding="utf-8"
    )

    return len(text.split())


# ============================================================
# SPLIT SCORE
# ============================================================

def calculate_score(
    current_words,
    target_words,
):
    """
    Measure how close a split is to its target.

    Lower score = better.
    """

    if target_words == 0:
        return 0

    return abs(
        current_words - target_words
    ) / target_words


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find processed books
    # --------------------------------------------------------

    files = sorted(
        PROCESSED_DIR.glob("*.txt")
    )

    if not files:

        print(
            f"No .txt files found in:\n"
            f"{PROCESSED_DIR}"
        )

        return

    print(
        f"Found {len(files)} processed books."
    )

    # --------------------------------------------------------
    # Calculate word counts
    # --------------------------------------------------------

    books = []

    for file_path in files:

        words = count_words(file_path)

        books.append({
            "file": file_path,
            "words": words
        })

    total_words = sum(
        book["words"]
        for book in books
    )

    # --------------------------------------------------------
    # Target word counts
    # --------------------------------------------------------

    target_train = (
        total_words * TRAIN_RATIO
    )

    target_validation = (
        total_words * VALIDATION_RATIO
    )

    target_test = (
        total_words * TEST_RATIO
    )

    print("\nCorpus:")
    print(
        f"  Books: {len(books)}"
    )
    print(
        f"  Words: {total_words:,}"
    )

    print("\nTarget distribution:")
    print(
        f"  Train:      {target_train:,.0f}"
    )
    print(
        f"  Validation: {target_validation:,.0f}"
    )
    print(
        f"  Test:       {target_test:,.0f}"
    )

    # --------------------------------------------------------
    # Shuffle deterministically
    # --------------------------------------------------------

    random.shuffle(books)

    # --------------------------------------------------------
    # Sort large books first
    #
    # This makes balancing easier because large books
    # have the biggest effect on the final distribution.
    # --------------------------------------------------------

    books.sort(
        key=lambda book: book["words"],
        reverse=True
    )

    # --------------------------------------------------------
    # Initialize splits
    # --------------------------------------------------------

    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    current_words = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    targets = {
        "train": target_train,
        "validation": target_validation,
        "test": target_test
    }

    # --------------------------------------------------------
    # Assign each complete book to the split that is currently
    # furthest from its target.
    # --------------------------------------------------------

    for book in books:

        # Calculate how much each split currently exceeds
        # or falls short of its target.

        candidate_scores = {}

        for split_name in splits:

            score = calculate_score(
                current_words[split_name],
                targets[split_name]
            )

            candidate_scores[split_name] = score

        # Choose the split with the largest relative deficit.
        #
        # deficit > 0 means the split is below its target.

        deficits = {}

        for split_name in splits:

            if targets[split_name] == 0:
                deficits[split_name] = 0
            else:
                deficits[split_name] = (
                    targets[split_name]
                    - current_words[split_name]
                ) / targets[split_name]

        # Prefer the split that is furthest below target.
        selected_split = max(
            deficits,
            key=deficits.get
        )

        splits[selected_split].append(
            book
        )

        current_words[selected_split] += (
            book["words"]
        )

    # --------------------------------------------------------
    # Display assignment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BOOK ASSIGNMENT")
    print("=" * 70)

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"\n{split_name.upper()}"
        )

        split_books = sorted(
            splits[split_name],
            key=lambda book: book["file"].name
        )

        for book in split_books:

            print(
                f"  {book['file'].name:<35}"
                f"{book['words']:>10,} words"
            )

    # --------------------------------------------------------
    # Create split files and manifest
    # --------------------------------------------------------

    manifest = {
        "seed": SEED,
        "total_books": len(books),
        "total_words": total_words,
        "target_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO
        },
        "train": [],
        "validation": [],
        "test": []
    }

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        split_books = sorted(
            splits[split_name],
            key=lambda book: book["file"].name
        )

        output_file = (
            OUTPUT_DIR /
            f"{split_name}.txt"
        )

        with output_file.open(
            "w",
            encoding="utf-8"
        ) as output:

            for book in split_books:

                text = book["file"].read_text(
                    encoding="utf-8"
                ).strip()

                if not text:
                    continue

                output.write(text)
                output.write("\n\n")

                manifest[split_name].append({
                    "file": book["file"].name,
                    "words": book["words"]
                })

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_DIR /
        "split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        split_words = current_words[
            split_name
        ]

        actual_ratio = (
            split_words / total_words
        )

        print(
            f"{split_name:<12}"
            f"Books: {len(splits[split_name]):>3}   "
            f"Words: {split_words:>10,}   "
            f"Share: {actual_ratio * 100:>6.2f}%"
        )

    print("\nFiles created:")

    print(
        f"  {OUTPUT_DIR / 'train.txt'}"
    )

    print(
        f"  {OUTPUT_DIR / 'validation.txt'}"
    )

    print(
        f"  {OUTPUT_DIR / 'test.txt'}"
    )

    print(
        f"  {manifest_path}"
    )


if __name__ == "__main__":
    main()