from pathlib import Path

import re
import unicodedata

from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# GUTENBERG MARKERS
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
# MOJIBAKE REPAIR
# ============================================================

def repair_mojibake(text):
    """
    Repair UTF-8 text that was incorrectly decoded
    using Windows-1252.
    """

    try:
        repaired = text.encode(
            "cp1252"
        ).decode(
            "utf-8"
        )

        return repaired

    except UnicodeError:

        return text

# ============================================================
# UNICODE NORMALIZATION
# ============================================================

def normalize_unicode(text):

    return unicodedata.normalize(
        "NFKC",
        text
    )


# ============================================================
# GUTENBERG WRAPPER REMOVAL
# ============================================================

def remove_gutenberg_boilerplate(text):

    lines = text.splitlines()

    start_index = 0
    end_index = len(lines)

    for i, line in enumerate(lines):

        if any(
            marker in line
            for marker in GUTENBERG_START_MARKERS
        ):

            start_index = i + 1
            break

    for i in range(
        start_index,
        len(lines)
    ):

        if any(
            marker in lines[i]
            for marker in GUTENBERG_END_MARKERS
        ):

            end_index = i
            break

    return "\n".join(
        lines[start_index:end_index]
    )


# ============================================================
# FRONT MATTER HELPERS
# ============================================================

def is_transcriber_note_start(line):

    normalized = (
        line.strip()
        .lower()
        .replace("’", "'")
    )

    return (
        normalized.startswith(
            "transcriber's note"
        )
        or normalized.startswith(
            "transcribers note"
        )
    )


def is_editor_note_start(line):

    normalized = (
        line.strip()
        .lower()
        .replace("’", "'")
    )

    return (
        normalized.startswith(
            "editor's note"
        )
        or normalized.startswith(
            "editors note"
        )
    )


def is_gutenberg_note(line):

    normalized = line.strip().lower()

    return (
        "note: project gutenberg" in normalized
        or "project gutenberg has an html version" in normalized
        or "project gutenberg e-text" in normalized
    )


def is_illustration_list(line):

    normalized = line.strip().lower()

    return (
        normalized == "list of illustrations"
        or normalized == "list of illustrations."
    )


def is_chapter_heading(line):

    line = line.strip()

    pattern = re.compile(
        r"^(chapter|chap\.)\s+"
        r"([ivxlcdm]+|\d+)\b",
        re.IGNORECASE
    )

    return bool(
        pattern.match(line)
    )


# ============================================================
# REMOVE TRANSCRIBER / EDITORIAL NOTES
# ============================================================

def remove_note_blocks(lines):

    cleaned = []

    skip_mode = False

    for i, line in enumerate(lines):

        stripped = line.strip()

        # ----------------------------------------------------
        # Start of Transcriber's Note
        # ----------------------------------------------------

        if is_transcriber_note_start(
            stripped
        ):

            skip_mode = True

            continue

        # ----------------------------------------------------
        # Start of Editor's Note
        # ----------------------------------------------------

        if is_editor_note_start(
            stripped
        ):

            skip_mode = True

            continue

        # ----------------------------------------------------
        # Gutenberg-specific note
        # ----------------------------------------------------

        if is_gutenberg_note(
            stripped
        ):

            skip_mode = True

            continue

        # ----------------------------------------------------
        # End note block when we reach a strong
        # book-content marker.
        # ----------------------------------------------------

        if skip_mode:

            if is_chapter_heading(
                stripped
            ):

                skip_mode = False

                cleaned.append(
                    line
                )

            continue

        cleaned.append(
            line
        )

    return cleaned


# ============================================================
# REMOVE KNOWN PUBLISHER MATERIAL
# ============================================================

def remove_publisher_front_matter(lines):

    cleaned = []

    skip_mode = False

    publisher_markers = [
        "books by the same author",
        "list of illustrations",
        "bibliophile edition",
        "university library association",
        "copyright",
        "andrew lang edition",
    ]

    for line in lines:

        normalized = (
            line.strip()
            .lower()
        )

        if any(
            marker in normalized
            for marker in publisher_markers
        ):

            skip_mode = True

            continue

        if skip_mode:

            # Stop skipping when actual chapter
            # content begins.

            if is_chapter_heading(
                line
            ):

                skip_mode = False

                cleaned.append(
                    line
                )

            continue

        cleaned.append(
            line
        )

    return cleaned


# ============================================================
# REMOVE OBVIOUS METADATA LINES
# ============================================================

def remove_metadata_lines(lines):

    metadata_patterns = [

        r"^\[illustration\]$",

        r"^copyright,?\s+\d{4}$",

        r"^by\s*$",

        r"^university library association$",

        r"^philadelphia$",

    ]

    compiled = [
        re.compile(
            pattern,
            re.IGNORECASE
        )
        for pattern in metadata_patterns
    ]

    cleaned = []

    for line in lines:

        stripped = line.strip()

        if any(
            pattern.match(stripped)
            for pattern in compiled
        ):

            continue

        cleaned.append(
            line
        )

    return cleaned


# ============================================================
# FRONT MATTER PIPELINE
# ============================================================

def remove_front_matter(text):

    lines = text.splitlines()

    original_count = len(lines)

    lines = remove_note_blocks(
        lines
    )

    lines = remove_publisher_front_matter(
        lines
    )

    lines = remove_metadata_lines(
        lines
    )

    removed = (
        original_count
        - len(lines)
    )

    print(
        f"  Metadata/front-matter lines removed: "
        f"{removed:,}"
    )

    return "\n".join(
        lines
    )


# ============================================================
# GENERAL CLEANING
# ============================================================

def clean_text(text):

    # Repair encoding corruption first
    text = repair_mojibake(
        text
    )

    text = normalize_unicode(
        text
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    return "\n".join(
        lines
    ).strip()


# ============================================================
# SENTENCE SEGMENTATION
# ============================================================

def segment_sentences(text):

    paragraphs = text.split(
        "\n"
    )

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_sentences = (
            sent_tokenize(
                paragraph
            )
        )

        for sentence in paragraph_sentences:

            sentence = sentence.strip()

            if sentence:

                sentences.append(
                    sentence
                )

    return sentences


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    sentences
):

    full_text = " ".join(
        sentences
    )

    words = word_tokenize(
        full_text
    )

    word_count = sum(
        1
        for token in words
        if re.search(
            r"[A-Za-z0-9]",
            token
        )
    )

    return {
        "characters": len(
            full_text
        ),
        "sentences": len(
            sentences
        ),
        "words": word_count,
    }


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    input_path,
    output_path
):

    print(
        f"\nProcessing: "
        f"{input_path.name}"
    )

    text = input_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    raw_characters = len(
        text
    )

    # --------------------------------------------------------
    # Gutenberg wrapper
    # --------------------------------------------------------

    text = remove_gutenberg_boilerplate(
        text
    )

    # --------------------------------------------------------
    # Front matter / metadata
    # --------------------------------------------------------

    text = remove_front_matter(
        text
    )

    # --------------------------------------------------------
    # General cleanup + encoding repair
    # --------------------------------------------------------

    text = clean_text(
        text
    )

    # --------------------------------------------------------
    # Sentence segmentation
    # --------------------------------------------------------

    sentences = segment_sentences(
        text
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_statistics(
        sentences
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_text = "\n".join(
        sentences
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        output_text,
        encoding="utf-8"
    )

    print(
        f"  Characters: "
        f"{raw_characters:,} → "
        f"{stats['characters']:,}"
    )

    print(
        f"  Sentences: "
        f"{stats['sentences']:,}"
    )

    print(
        f"  Words: "
        f"{stats['words']:,}"
    )

    print(
        f"  Output: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        RAW_DIR.glob("*.txt")
    )

    if not files:

        print(
            f"No .txt files found in: "
            f"{RAW_DIR}"
        )

        return

    print(
        f"Found {len(files)} "
        f"Gutenberg text file(s)."
    )

    total_sentences = 0
    total_words = 0

    for input_path in files:

        output_path = (
            PROCESSED_DIR
            / input_path.name
        )

        process_file(
            input_path,
            output_path
        )

        processed_text = (
            output_path.read_text(
                encoding="utf-8"
            )
        )

        sentences = [
            line
            for line
            in processed_text.splitlines()
            if line.strip()
        ]

        stats = calculate_statistics(
            sentences
        )

        total_sentences += (
            stats["sentences"]
        )

        total_words += (
            stats["words"]
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "GUTENBERG PROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Books: "
        f"{len(files):,}"
    )

    print(
        f"Sentences: "
        f"{total_sentences:,}"
    )

    print(
        f"Words: "
        f"{total_words:,}"
    )

    print(
        f"Output: "
        f"{PROCESSED_DIR}"
    )


if __name__ == "__main__":

    main()