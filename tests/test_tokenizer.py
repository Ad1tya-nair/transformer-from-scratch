from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_PATH = Path(
    r"C:\Users\Aditya N\Downloads\llm\tokenizer\tokenizer.json"
)


def test_sentence(tokenizer, text):

    encoding = tokenizer.encode(text)

    print("\nTEXT:")
    print(text)

    print("\nTOKENS:")
    print(encoding.tokens)

    print("\nTOKEN IDS:")
    print(encoding.ids)

    print(
        f"\nNumber of tokens: "
        f"{len(encoding.ids)}"
    )


def main():

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    print(
        "Vocabulary size:",
        tokenizer.get_vocab_size()
    )

    test_sentences = [

        "The professor gave me a cold response.",

        "The water was extremely cold.",

        "The detective entered the room quietly.",

        "She was unhappy about the situation.",

        "The professor was delighted with the result.",

        "unhappiness",

        "professor",

        "coldness",

        "can't",

        "Mr. Darcy"

    ]

    for sentence in test_sentences:
        test_sentence(
            tokenizer,
            sentence
        )


if __name__ == "__main__":
    main()