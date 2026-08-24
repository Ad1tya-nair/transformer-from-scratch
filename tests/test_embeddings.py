import torch

from embeddings import TokenEmbedding


VOCAB_SIZE = 8000
EMBEDDING_DIM = 256
BATCH_SIZE = 32
CONTEXT_LENGTH = 128


def main():

    embedding = TokenEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM
    )

    # Simulate the output from our DataLoader
    token_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, CONTEXT_LENGTH)
    )

    print(
        "Token IDs:",
        token_ids.shape
    )

    vectors = embedding(token_ids)

    print(
        "Embeddings:",
        vectors.shape
    )

    parameters = sum(
        p.numel()
        for p in embedding.parameters()
    )

    print(
        "Embedding parameters:",
        parameters
    )


if __name__ == "__main__":
    main()