import random

import torch

from tokenizers import Tokenizer

from src.model.language_model import LanguageModel
from src.checkpoint import load_checkpoint


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

VOCAB_SIZE = 8000

CONTEXT_LENGTH = 128

EMBEDDING_DIM = 256

NUM_HEADS = 4

FEED_FORWARD_DIM = 1024

NUM_LAYERS = 4

TOKENIZER_FILE = (
    "tokenizer/tokenizer.json"
)

CHECKPOINT_PATH = (
    "checkpoints_fresh/checkpoint_epoch_6.pt"
)


# --------------------------------------------------
# SAMPLING OPTIONS
# --------------------------------------------------

TEMPERATURE_OPTIONS = .80

TOP_K_OPTIONS = [
    60,70,45,50,80,85
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model(
    checkpoint_path,
    device
):

    model = LanguageModel(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        device=device
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    return model


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def encode_text(
    text,
    tokenizer
):

    encoded = tokenizer.encode(
        text
    )

    return torch.tensor(
        encoded.ids,
        dtype=torch.long
    )


# --------------------------------------------------
# NEXT TOKEN PREDICTION
# --------------------------------------------------

def predict_next_token(
    model,
    token_ids,
    device,
    temperature,
    top_k
):

    # Add batch dimension
    input_ids = (
        token_ids
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            input_ids
        )

    # --------------------------------------------------
    # FINAL POSITION
    # --------------------------------------------------

    next_token_logits = (
        logits[:, -1, :]
    )

    # --------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------

    next_token_logits = (
        next_token_logits /
        temperature
    )

    # --------------------------------------------------
    # TOP-K FILTERING
    # --------------------------------------------------

    top_k = min(
        top_k,
        next_token_logits.size(-1)
    )

    values, _ = torch.topk(
        next_token_logits,
        top_k
    )

    minimum_value = (
        values[:, -1].unsqueeze(-1)
    )

    next_token_logits = torch.where(
        next_token_logits
        < minimum_value,

        torch.full_like(
            next_token_logits,
            float("-inf")
        ),

        next_token_logits
    )

    # --------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1
    )

    # --------------------------------------------------
    # RANDOM SAMPLING
    # --------------------------------------------------

    next_token_id = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token_id.item()


# --------------------------------------------------
# AUTOREGRESSIVE GENERATION
# --------------------------------------------------

def generate(
    model,
    token_ids,
    device,
    max_new_tokens,
    temperature,
    top_k,
    eos_token_id=None
):

    generated_ids = (
        token_ids.clone()
    )

    for _ in range(
        max_new_tokens
    ):

        # --------------------------------------------------
        # CONTEXT WINDOW
        # --------------------------------------------------

        input_ids = generated_ids[
            -CONTEXT_LENGTH:
        ]

        # --------------------------------------------------
        # PREDICT NEXT TOKEN
        # --------------------------------------------------

        next_token_id = predict_next_token(
            model=model,
            token_ids=input_ids,
            device=device,
            temperature=temperature,
            top_k=top_k
        )

        # --------------------------------------------------
        # APPEND TOKEN
        # --------------------------------------------------

        next_token = torch.tensor(
            [next_token_id],
            dtype=torch.long
        )

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token
            ]
        )

        # --------------------------------------------------
        # EOS CHECK
        # --------------------------------------------------

        if (
            eos_token_id is not None
            and
            next_token_id == eos_token_id
        ):

            break

    return generated_ids


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = load_model(
        CHECKPOINT_PATH,
        device
    )

    # --------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------

    tokenizer = Tokenizer.from_file(
        TOKENIZER_FILE
    )

    print(
        f"\nTokenizer vocabulary: "
        f"{tokenizer.get_vocab_size():,}"
    )

    # --------------------------------------------------
    # EOS TOKEN
    # --------------------------------------------------

    eos_token_id = (
        tokenizer.token_to_id(
            "[EOS]"
        )
    )

    # --------------------------------------------------
    # INTERACTIVE GENERATION LOOP
    # --------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "INTERACTIVE TEXT GENERATION"
    )

    print(
        "=" * 60
    )

    print(
        "\nSampling is enabled."
    )

    print(
        "Press Ctrl+C to exit."
    )

    print(
        "=" * 60
    )

    while True:

        # --------------------------------------------------
        # PROMPT
        # --------------------------------------------------

        text = input(
            "\nEnter prompt: "
        )

        if not text.strip():

            print(
                "Prompt cannot be empty."
            )

            continue

        # --------------------------------------------------
        # TOKEN LIMIT
        # --------------------------------------------------

        while True:

            try:

                max_new_tokens = int(
                    input(
                        "Enter number of tokens "
                        "to generate: "
                    )
                )

                if max_new_tokens <= 0:

                    print(
                        "Please enter a positive "
                        "integer."
                    )

                    continue

                break

            except ValueError:

                print(
                    "Please enter a valid "
                    "integer."
                )

        # --------------------------------------------------
        # RANDOM SAMPLING PARAMETERS
        # --------------------------------------------------

        temperature = TEMPERATURE_OPTIONS

        top_k = random.choice(
            TOP_K_OPTIONS
        )

        print(
            f"\nSampling parameters:"
        )

        print(
            f"Temperature: {temperature:.2f}"
        )

        print(
            f"Top-k: {top_k}"
        )

        # --------------------------------------------------
        # TOKENIZE
        # --------------------------------------------------

        token_ids = encode_text(
            text,
            tokenizer
        )

        print(
            f"Input tokens: "
            f"{token_ids.shape[0]}"
        )

        # --------------------------------------------------
        # GENERATE
        # --------------------------------------------------

        generated_ids = generate(
            model=model,
            token_ids=token_ids,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id
        )

        # --------------------------------------------------
        # DECODE
        # --------------------------------------------------

        generated_text = tokenizer.decode(
            generated_ids.tolist()
        )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "GENERATED TEXT"
        )

        print(
            "=" * 60
        )

        print(
            generated_text
        )

        print(
            "=" * 60
        )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nGeneration stopped."
        )