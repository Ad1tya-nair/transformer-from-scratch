from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 8_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.txt"
)

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
)

TOKENIZER_FILE = (
    TOKENIZER_DIR
    / "tokenizer.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("BPE TOKENIZER TRAINING")
    print("=" * 60)

    print(f"Training corpus:")
    print(f"  {TRAIN_FILE}")

    print(f"\nTarget vocabulary: {VOCAB_SIZE:,}")

    # --------------------------------------------------------
    # Create empty BPE tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]"
        )
    )

    # Split text into words before BPE learns subwords
    tokenizer.pre_tokenizer = Whitespace()

    # --------------------------------------------------------
    # Configure trainer
    # --------------------------------------------------------

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,

        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]"
        ],

        min_frequency=2
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    tokenizer.train(
        files=[
            str(TRAIN_FILE)
        ],
        trainer=trainer
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    tokenizer.save(
        str(TOKENIZER_FILE)
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    actual_vocab_size = (
        tokenizer.get_vocab_size()
    )

    print("\nTokenizer training complete.")

    print(
        f"Actual vocabulary size: "
        f"{actual_vocab_size:,}"
    )

    print(
        f"Saved to:\n"
        f"{TOKENIZER_FILE}"
    )


if __name__ == "__main__":
    main()