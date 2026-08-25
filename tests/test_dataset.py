from data import LanguageModelDataset


def main():

    dataset = LanguageModelDataset(
        split="train"
    )

    print(
        "Number of sequences:",
        len(dataset)
    )

    x, y = dataset[0]

    print(
        "\nInput shape:",
        x.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    print(
        "\nFirst 20 input IDs:"
    )

    print(
        x[:20]
    )

    print(
        "\nFirst 20 target IDs:"
    )

    print(
        y[:20]
    )

    print(
        "\nShift verification:"
    )

    print(
        "Input[1:] == Target[:-1]:",
        bool(
            (x[1:] == y[:-1]).all()
        )
    )


if __name__ == "__main__":
    main()