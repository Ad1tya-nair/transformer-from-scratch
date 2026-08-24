from pathlib import Path
import re
import unicodedata

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# GUTENBERG HEADER / FOOTER
# ============================================================

GUTENBERG_START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
]

GUTENBERG_END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_unicode(text):
    """
    Normalize Unicode characters into a consistent representation.
    """
    return unicodedata.normalize("NFKC", text)


def remove_gutenberg_boilerplate(text):
    """
    Remove Project Gutenberg header and footer.
    """

    lines = text.splitlines()

    start_index = 0
    end_index = len(lines)

    # Find beginning of actual book
    for i, line in enumerate(lines):
        if any(marker in line for marker in GUTENBERG_START_MARKERS):
            start_index = i + 1
            break

    # Find end of actual book
    for i in range(start_index, len(lines)):
        if any(marker in lines[i] for marker in GUTENBERG_END_MARKERS):
            end_index = i
            break

    return "\n".join(lines[start_index:end_index])


def clean_text(text):
    """
    General text normalization after Gutenberg boilerplate removal.
    """

    text = normalize_unicode(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip whitespace from each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines
    lines = [line for line in lines if line]

    text = "\n".join(lines)

    return text.strip()


# ============================================================
# SENTENCE SEGMENTATION
# ============================================================

def segment_sentences(text):
    """
    Split cleaned text into sentences.

    Each sentence is written on its own line.
    """

    paragraphs = text.split("\n")

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_sentences = sent_tokenize(paragraph)

        for sentence in paragraph_sentences:

            sentence = sentence.strip()

            if sentence:
                sentences.append(sentence)

    return sentences


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(sentences):
    """
    Calculate basic corpus statistics.
    """

    full_text = " ".join(sentences)

    words = word_tokenize(full_text)

    # Count alphabetic/numeric tokens as words
    word_count = sum(
        1 for token in words
        if re.search(r"[A-Za-z0-9]", token)
    )

    return {
        "characters": len(full_text),
        "sentences": len(sentences),
        "words": word_count,
    }


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_path, output_path):

    print(f"\nProcessing: {input_path.name}")

    # Read raw Gutenberg file
    text = input_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    raw_characters = len(text)

    # Gutenberg cleanup
    text = remove_gutenberg_boilerplate(text)

    # General cleanup
    text = clean_text(text)

    # Sentence segmentation
    sentences = segment_sentences(text)

    # Statistics
    stats = calculate_statistics(sentences)

    # Save one sentence per line
    output_text = "\n".join(sentences)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        output_text,
        encoding="utf-8"
    )

    print(f"  Characters: {raw_characters:,} → {stats['characters']:,}")
    print(f"  Sentences:  {stats['sentences']:,}")
    print(f"  Words:      {stats['words']:,}")
    print(f"  Output:     {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    # Make sure output directory exists
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(RAW_DIR.glob("*.txt"))

    if not files:
        print(f"No .txt files found in: {RAW_DIR}")
        return

    print(f"Found {len(files)} Gutenberg text file(s).")

    total_sentences = 0
    total_words = 0

    for input_path in files:

        output_path = PROCESSED_DIR / input_path.name

        process_file(
            input_path,
            output_path
        )

        # Read the generated file to accumulate totals
        processed_text = output_path.read_text(
            encoding="utf-8"
        )

        sentences = [
            line
            for line in processed_text.splitlines()
            if line.strip()
        ]

        stats = calculate_statistics(sentences)

        total_sentences += stats["sentences"]
        total_words += stats["words"]

    print("\n" + "=" * 60)
    print("GUTENBERG PROCESSING COMPLETE")
    print("=" * 60)

    print(f"Books:      {len(files):,}")
    print(f"Sentences:  {total_sentences:,}")
    print(f"Words:      {total_words:,}")
    print(f"Output:     {PROCESSED_DIR}")


if __name__ == "__main__":
    main()