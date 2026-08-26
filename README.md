# Building a Transformer Language Model from Scratch

A learning-focused implementation of a small Transformer-based language model built from scratch using PyTorch.

The main purpose of this project is not to build a production-ready LLM, but to understand **how modern language models work internally by implementing the major components myself**.

Rather than relying on a high-level Transformer implementation, the project gradually builds the architecture from individual components such as tokenization, embeddings, self-attention, multi-head attention, feed-forward networks, residual connections, Transformer blocks, and the language-model head.

---

## Why I Built This

Large language models can be easy to use but difficult to understand.

Libraries such as Hugging Face Transformers make it possible to load and train sophisticated models with very little code. While that is extremely useful in practice, it can hide many of the mechanisms that make a Transformer work.

This project takes the opposite approach:

> **Build a small language model and understand what every major component is doing.**

The project is being developed alongside notes and experiments covering the architecture step by step.

The focus is therefore on:

- Learning
- Experimentation
- Understanding the architecture and underlying concepts
- Implementing components independently
- Observing how training affects model behaviour
- Experimenting with text generation

---

# What This Project Covers

The project currently covers the basic pipeline of a Transformer language model:

```text
Raw Text
   │
   ▼
Text Cleaning
   │
   ▼
Train / Validation / Test Split
   │
   ▼
BPE Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Token Embeddings
   │
   ▼
Positional Embeddings
   │
   ▼
Transformer Blocks
   │
   ├── Multi-Head Self-Attention
   ├── Residual Connections
   ├── Layer Normalization
   └── Feed-Forward Network
   │
   ▼
Language Model Head
   │
   ▼
Vocabulary Logits
   │
   ▼
Sampling
   │
   ├── Temperature
   └── Top-k
   │
   ▼
Generated Text
```

# Model Architecture

The current model is intentionally small enough to train on a consumer GPU while still containing the major components of a decoder-style Transformer language model.

| Component | Configuration |
|---|---:|
| Vocabulary | 8,000 tokens |
| Context length | 128 tokens |
| Embedding dimension | 256 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Feed-forward dimension | 1,024 |

The model is implemented as separate Python modules rather than as one large class. This makes it easier to study each component independently and understand how they work together.

---

## Token Embeddings

After tokenization, token IDs are converted into dense numerical vectors.

```text
Token ID
   │
   ▼
Embedding Table
   │
   ▼
Dense Vector
```

The embedding table is learned during training.

Instead of representing a token simply as an integer such as:

```text
297
```

the model represents it as a 256-dimensional vector.

These learned vectors allow the model to develop useful representations of different tokens.

---

## Positional Embeddings

Self-attention does not inherently understand the order in which tokens appear.

For example:

```text
The dog chased the cat
```

and:

```text
The cat chased the dog
```

contain the same words but have completely different meanings.

Positional information is therefore added to the token representations so that the model can distinguish between different positions in the sequence.

Conceptually:

```text
Token Embedding
       +
Positional Embedding
       │
       ▼
Transformer Input
```

---

## Self-Attention

Self-attention allows each token to consider other tokens within its context.

The basic mechanism uses three learned representations:

- Query
- Key
- Value

Conceptually:

```text
Input
  │
  ├──────► Query
  ├──────► Key
  └──────► Value
             │
             ▼
        Attention Scores
             │
             ▼
      Weighted Values
             │
             ▼
      Context-aware Output
```

This allows the representation of a token to depend on other tokens in the sequence.

For example, in a sentence such as:

> The animal didn't cross the road because it was tired.

different attention patterns can allow the model to associate words with relevant parts of the surrounding context.

The important idea is that a token's representation can be influenced by other tokens in the sequence.

Understanding this mechanism is one of the main reasons for building the project from scratch.

---

## Multi-Head Attention

Instead of performing a single attention operation, the model uses multiple attention heads.

The current configuration uses:

```text
4 attention heads
```

Each head performs its own attention operation and can learn different relationships between tokens.

Conceptually:

```text
                    Input
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Head 1       Head 2      ... Head 4
          │           │              │
          ▼           ▼              ▼
       Attention    Attention      Attention
          │           │              │
          └───────────┼──────────────┘
                      │
                      ▼
                  Concatenate
                      │
                      ▼
                Linear Projection
```

Using multiple heads allows the model to examine the same sequence from several learned perspectives.

---

## Feed-Forward Network

After self-attention, the representations pass through a feed-forward network.

The current feed-forward network uses:

```text
256
 │
 ▼
1024
 │
 ▼
256
```

The first layer expands the representation from 256 dimensions to 1,024 dimensions.

A nonlinear activation is then applied before projecting the representation back down to 256 dimensions.

Conceptually:

```text
Transformer Representation
          │
          ▼
       Linear
     256 → 1024
          │
          ▼
    Non-linear Activation
          │
          ▼
       Linear
     1024 → 256
          │
          ▼
        Output
```

The attention mechanism allows tokens to interact with one another, while the feed-forward network performs additional transformations on each token's representation.

---

## Residual Connections and Layer Normalization

Transformer blocks use residual connections to help information and gradients flow through the network.

A simplified view of the block is:

```text
                 Input
                   │
          ┌────────┴────────┐
          │                 │
          │          Multi-Head
          │           Attention
          │                 │
          │                 ▼
          └──────────────► Add
                            │
                       LayerNorm
                            │
                            ▼
                           FFN
                            │
          ┌─────────────────┘
          │
          ▼
         Add
          │
      LayerNorm
          │
          ▼
        Output
```

The residual connection allows the original representation to be combined with the transformed representation.

Layer normalization helps keep the activations in a stable range during training.

---

## Transformer Block

The components above are combined into a Transformer block.

A single block can be viewed as:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Output
```

The current model contains:

```text
4 Transformer blocks
```

These blocks are stacked sequentially, allowing the model to build increasingly complex representations.

```text
Input
  │
  ▼
Transformer Block 1
  │
  ▼
Transformer Block 2
  │
  ▼
Transformer Block 3
  │
  ▼
Transformer Block 4
  │
  ▼
Output Representation
```

---

## Language Model Head

After passing through the Transformer blocks, the resulting representations are converted into scores for every token in the vocabulary.

The current vocabulary contains:

```text
8,000 tokens
```

Therefore, for each position, the language model head produces:

```text
256-dimensional representation
            │
            ▼
        LM Head
            │
            ▼
       8,000 logits
```

These logits represent the model's raw scores for the possible next tokens.

They are converted into probabilities during generation.

```text
Logits
  │
  ▼
Softmax
  │
  ▼
Probability distribution
  │
  ▼
Next token
```

The model is trained to predict the next token given the previous tokens in the context.

---

# Tokenization

The project uses a **Byte Pair Encoding (BPE)** tokenizer.

The tokenizer currently contains:

```text
8,000 tokens
```

The tokenizer is trained on the training corpus before the text is converted into the binary dataset used for model training.

A sentence such as:

```text
The train arrived
```

is converted into token IDs such as:

```text
[297, 1827, 3132]
```

The exact IDs depend on the trained tokenizer vocabulary.

The important transformation is:

```text
Text
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
```

The tokenizer is kept separate from the language model so that tokenization can be studied independently from the neural network.

---

# Dataset and Preprocessing

The training corpus consists primarily of public-domain literary texts obtained from Project Gutenberg.

The preprocessing pipeline is:

```text
Raw Gutenberg Books
        │
        ▼
Gutenberg Cleanup
        │
        ▼
Text Normalization
        │
        ▼
Sentence Segmentation
        │
        ▼
Processed Books
        │
        ▼
Train / Validation / Test Split
        │
        ▼
BPE Tokenization
        │
        ▼
Binary Token Dataset
        │
        ▼
PyTorch Dataset
```

The project separates the raw, processed, split, and tokenized representations of the dataset.

Generated datasets are intentionally kept out of version control.

---

# Next-Token Prediction

The model is trained using the standard language-modeling objective of predicting the next token.

For a sequence such as:

```text
The train arrived at
```

the model learns to predict the token following each position.

Conceptually:

```text
Input:
The train arrived

Target:
train arrived ...
```

More generally:

```text
x₁ x₂ x₃ ... xₙ
        │
        ▼
   Transformer
        │
        ▼
Predict next token
```

During training, the input and target sequences are shifted by one position.

This allows the model to learn the probability:

```text
P(next token | previous tokens)
```

The training objective is cross-entropy loss between the predicted probability distribution and the actual next token.

---

# Training

The model is trained from scratch rather than starting from pretrained Transformer weights.

Current training configuration:

| Parameter | Value |
|---|---:|
| Batch size | 32 |
| Context length | 128 |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Epochs in fresh experiment | 10 |

Training is performed using PyTorch.

The training pipeline includes:

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
Transformer
   │
   ▼
Logits
   │
   ▼
Cross-Entropy Loss
   │
   ▼
Backpropagation
   │
   ▼
AdamW
   │
   ▼
Updated Parameters
```

Validation loss is calculated after each epoch to monitor how well the model generalizes beyond the training examples.

---

# Checkpointing

The training system saves checkpoints containing:

- Model parameters
- Optimizer state
- Epoch number
- Training loss
- Validation loss

This allows training to be resumed rather than starting over after every interruption.

It also makes it possible to compare different stages of training.

For example:

```text
checkpoint_epoch_1.pt
checkpoint_epoch_2.pt
...
checkpoint_epoch_10.pt
```

Generated checkpoint files are kept outside version control because of their size.

---

# Training Experiment

One of the experiments involved retraining the model from scratch after expanding the literary corpus.

The fresh training run produced the following results:

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 5.5380 | 5.4444 |
| 2 | 4.7517 | 5.2104 |
| 3 | 4.4855 | 5.1327 |
| 4 | 4.3243 | 5.0743 |
| 5 | 4.2103 | 5.0473 |
| 6 | 4.1231 | **5.0310** |
| 7 | 4.0533 | 5.0403 |
| 8 | 3.9952 | 5.0337 |
| 9 | 3.9455 | 5.0432 |
| 10 | 3.9022 | 5.0347 |

The lowest validation loss in this experiment occurred at **epoch 6**.

One useful observation from this experiment was that training loss continued to decrease after epoch 6, while validation loss stopped improving significantly.

This provided a practical example of why validation data is important when training neural networks.

---

# Text Generation

Once trained, the model can generate text autoregressively.

The generation process is:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
Transformer
  │
  ▼
Next-token probabilities
  │
  ▼
Sampling
  │
  ▼
Append selected token
  │
  ▼
Repeat
```

The generation script allows the user to provide:

- A text prompt
- The maximum number of new tokens to generate

The generated sequence is then decoded back into text using the same tokenizer.

---

# Sampling

The model does not always select the highest-probability token.

Instead, generation uses sampling so that multiple plausible continuations can be produced.

Two important sampling parameters are **temperature** and **Top-k**.

## Temperature

Temperature controls the sharpness of the probability distribution.

Conceptually:

```text
Adjusted logits = logits / temperature
```

A lower temperature makes the distribution more concentrated around high-probability tokens.

A higher temperature makes the distribution flatter and gives lower-probability tokens a greater chance of being selected.

Simplified:

```text
Lower temperature
      │
      ▼
More predictable output

Higher temperature
      │
      ▼
More varied output
```

---

## Top-k Sampling

Top-k sampling restricts the candidate tokens to the `k` highest-probability tokens.

For example:

```text
Top-k = 20
```

means that only the 20 most likely candidate tokens are considered during sampling.

Conceptually:

```text
Full vocabulary
      │
      ▼
Rank by probability
      │
      ▼
Keep top k tokens
      │
      ▼
Sample from candidates
```

This prevents very unlikely tokens from being selected while still allowing some randomness.

During experimentation, a temperature around `0.80` and a Top-k value around `20` produced relatively sensible outputs for the current model.

These values are empirical observations for this particular model rather than universally optimal settings.

---

# Project Structure

```text
llm/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── tokenizer/
│   └── tokenizer.json
│
├── src/
│   ├── model/
│   │   ├── attention.py
│   │   ├── attention_block.py
│   │   ├── embeddings.py
│   │   ├── feed_forward.py
│   │   ├── language_model.py
│   │   ├── lm_head.py
│   │   ├── positional.py
│   │   ├── transformer.py
│   │   └── transformer_block.py
│   │
│   ├── preprocessing/
│   │   └── gutenberg_parser.py
│   │
│   ├── checkpoint.py
│   ├── dataset.py
│   ├── dataset_split.py
│   ├── generate.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── prepare_dataset.py
│   ├── train.py
│   └── train_tokenizer.py
│
├── tests/
│
├── notes/
│
└── README.md
```

Generated datasets and model checkpoints are kept outside version control.

---

# Tests

The project contains tests for individual components as well as parts of the training pipeline.

Examples include:

```text
tests/
├── test_attention.py
├── test_attention_block.py
├── test_embeddings.py
├── test_feed_forward.py
├── test_language_model.py
├── test_lm_head.py
├── test_positional.py
├── test_tokenizer.py
├── test_transformer.py
├── test_transformer_block.py
├── test_training_loop.py
└── test_training_step.py
```

These tests are primarily intended to verify that individual components behave as expected during development.

---

# Learning Notes

A major part of this repository is the collection of notes created while building the model.

Topics currently covered include:

- Token IDs
- Tokenization
- Embeddings
- Positional embeddings
- Self-attention
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Transformer blocks
- Language-model heads
- Text generation

The notes focus on understanding **why each component exists, what happens to the data as it passes through the model, and how the different components fit together**.

---

# Running the Project

The general workflow is:

```text
1. Prepare raw text
2. Process Gutenberg books
3. Split the dataset
4. Train the tokenizer
5. Tokenize the dataset
6. Test the Dataset/DataLoader
7. Train the model
8. Evaluate checkpoints
9. Generate text
```

Run commands from the project root.

For example:

```bash
python -m src.train
```

and:

```bash
python -m src.generate
```

Individual test modules can also be run during development.

---

# Current Limitations

This is a **small educational language model**, not a competitive or production-ready LLM.

Current limitations include:

- Small vocabulary
- Short context window
- Relatively small model
- Limited training corpus
- Limited training compute
- No instruction tuning
- No RLHF
- No large-scale pretraining
- No sophisticated benchmark evaluation
- Generated text can become repetitive or incoherent

These limitations are acceptable for the purpose of the project.

The goal is to keep the model small enough that its architecture and behaviour can actually be studied and experimented with.

---


# Disclaimer

This repository is primarily a **learning and experimentation project**.

It is not intended to reproduce the scale or capabilities of modern production language models.

The main objective is to understand the components behind Transformer-based language models by implementing and training a small one from scratch.
