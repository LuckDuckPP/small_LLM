# Project Architecture and Documentation

## 1. Project Overview

The project is organized into two main parts:

1. **`UI/`** — Handles the user interface, chat workflow, model training controls, and interaction with the model.
2. **Main project folder** — Contains the language model implementation and the underlying Transformer architecture.

---

## 2. Project Structure

```text
project/
├── UI/
│   ├── engine.py
│   ├── interface.py
│   └── gui.py
│
|──main/
|   ├── model.py
|   ├── transformer.py
```

---

## 3. User Interface (`UI/`)

The `UI/` folder contains the files responsible for connecting the model to the chat interface and providing the user-facing application.

### 3.1 `engine.py` — Model and Chat Engine

`engine.py` acts as the binding layer between the chat application and the decoder-only Transformer model. It builds on `transformer.py` and `model.py`.

Its responsibilities include:

#### Tokenization and Conversation Formatting

- **`ByteTokenizer`** — A dependency-free tokenizer that uses 256 byte tokens along with chat-specific special tokens.
- **Chat template** — Converts conversations into token IDs using a structured format, for example:
  ```text
  <|user|>...<|eos|><|assistant|>
  ```

#### Chat Session Management

- **`ChatSession`** — Manages conversation history and supports saving and loading chats in JSON and Markdown formats.

#### Dataset Handling and Training

- **Datasets** — Supports:
  - Chat datasets in JSONL format, with loss calculated only on assistant replies.
  - Raw-text datasets.
- **`train_model`** — Handles model training using:
  - AdamW optimizer.
  - Learning-rate warm-up.
  - Cosine learning-rate schedule.
  - Progress callbacks.
  - A stop flag for cancelling training.

#### Text Generation

- **`stream_tokens`** — Generates tokens one at a time using a KV cache. Supports:
  - Temperature.
  - Top-k sampling.
  - Top-p sampling.
  - Repetition penalty.
  - Cancellation.

- **`ChatEngine`** — Builds the model prompt, removes older conversation turns when the context window is full, and streams decoded text chunks.

#### Model Utilities

- `save/load_checkpoint`
- `build_model`
- `SAMPLE_PAIRS` — Built-in demonstration dialogues.

### 3.2 `interface.py` — Application Entry Point

`interface.py` provides the entry point for launching the chat application.

Run this file first to start the interface.

### 3.3 `gui.py` — Graphical User Interface

`gui.py` implements the graphical window opened by `interface.py`.

Its features include:

#### Chat Interaction

- Streams generated replies token by token.
- Runs generation in a background thread to prevent the UI from freezing.
- Provides a Stop button and `Esc` key cancellation.
- Supports reply regeneration.
- Allows copying the most recent reply.

#### Generation Controls

Provides live controls for:

- Temperature.
- Top-k.
- Top-p.
- Maximum generation tokens.
- Repetition penalty.
- Random seed.

Also includes one-click generation presets:

- **Precise**
- **Balanced**
- **Creative**

#### Conversation and Context Management

- Editable system prompt.
- Context meter showing how much of the model's context window is being used.
- Automatically removes the oldest conversation turns when the context window is full.

#### Model Training

Provides a training window that supports:

- Training a model from scratch.
- Continuing existing training.
- Training on built-in sample dialogues.
- Loading chat datasets from `.jsonl` and `.json` files.
- Loading plain-text datasets from `.txt` files.
- Displaying a live training-loss curve.

#### File and Appearance Management

- Load and save model checkpoints.
- Load and save conversations in JSON or Markdown format.
- Light and dark themes.

---

## 4. Model and Transformer Architecture

The main project folder contains the language model implementation and the underlying Transformer architecture.

### 4.1 `model.py` — Decoder-Only Language Model

`model.py` implements the decoder-only language model used for text generation.

It connects the token embeddings, Transformer decoder, and language-model output head.

#### Model Components and Data Flow

```text
Token IDs
    |
    v
Token Embedding
    |
    v
Transformer Decoder
    |
    v
Language Model Head
    |
    v
Logits over Vocabulary
```

The responsibilities are divided as follows:

- **Token Embedding** — Converts input token IDs into vector representations.
- **Transformer Decoder** — Processes the embedded sequence and captures contextual information. Its architecture is implemented in `transformer.py`.
- **Language Model Head (LM Head)** — Projects the decoder output into vocabulary-sized logits.

#### Next-Token Prediction

The model uses its output logits to predict the next token.

**During training:**

- Uses cross-entropy loss.
- Compares the logits at each position `t` with the token that follows that position.

**During inference:**

- Generates text autoregressively, predicting one token at a time.
- Supports temperature, top-k, top-p, and repetition-penalty sampling.
- Handles end-of-sequence (EOS) tokens.
- Uses a KV cache to accelerate generation.

### 4.2 `transformer.py` — Transformer Architecture

`transformer.py` implements a modern decoder-only Transformer stack inspired by the LLaMA architecture, using pure PyTorch.

This file contains the Transformer architecture (the model's computational body). It does not contain the token embedding layer or the language-model output head.

#### Building Blocks

| Component | Responsibility |
|---|---|
| `TransformerConfig` | Dataclass containing the model's hyperparameters. |
| `RMSNorm` | Normalization layer used as a computationally efficient alternative to LayerNorm. |
| `RotaryEmbedding` | Implements rotary positional embeddings (RoPE) with cache awareness. |
| `KVCache` | Pre-allocated key/value cache for faster autoregressive generation. |
| `CausalSelfAttention` | Implements grouped-query attention (GQA), RoPE, causal masking, and KV caching. Uses PyTorch's fused scaled dot-product attention (SDPA) kernel. |
| `SwiGLUFeedForward` | Implements the gated SwiGLU feed-forward network. |
| `TransformerBlock` | Pre-normalization residual block containing attention and feed-forward sublayers. |
| `TransformerDecoder` | Stacks Transformer blocks and applies the final normalization layer. |

#### Transformer Block Data Flow

Each Transformer block uses a pre-normalization residual structure:

```text
Input (x)
   |
   +------------------------------+
   |                              |
   v                              |
 RMSNorm                          |
   |                              |
   v                              |
 Causal Self-Attention             |
 (RoPE + GQA + causal masking)     |
   |                              |
   +------------ Residual Add <---+
                  |
                  v
             Hidden State (h)
                  |
   +------------------------------+
   |                              |
   v                              |
 RMSNorm                          |
   |                              |
   v                              |
 SwiGLU Feed-Forward Network       |
   |                              |
   +------------ Residual Add <---+
                  |
                  v
              Block Output
```

---

## 5. End-to-End Execution Flow

The overall application flow can be summarized as follows:

```text
User
 |
 v
interface.py
 |
 v
gui.py
 |
 v
engine.py
 |
 +----------------------+
 |                      |
 v                      v
model.py             ChatSession
 |
 v
Token Embedding
 |
 v
TransformerDecoder
 |
 v
Language Model Head
 |
 v
Next-Token Logits
 |
 v
Token Sampling and KV Cache
 |
 v
Decoded Text Stream
 |
 v
GUI Chat Window
```

The UI layer manages conversation input and output, while `engine.py` coordinates tokenization, prompt construction, model execution, and text streaming. The model and Transformer files perform the actual neural-network computation.

---

## 6. File Responsibility Summary

| File | Main responsibility |
|---|---|
| `UI/engine.py` | Tokenization, chat sessions, dataset handling, training coordination, prompt construction, and text generation. |
| `UI/interface.py` | Entry point for launching the application. |
| `UI/gui.py` | Graphical chat interface, generation controls, training window, and file operations. |
| `model.py` | Token embeddings, Transformer integration, language-model head, and next-token prediction. |
| `transformer.py` | Transformer building blocks, attention, normalization, feed-forward layers, and decoder stack. |

---

## 7. Important Architecture Note

The original description of `transformer.py` refers to `model.py` as the wrapper containing the embedding and language-model head. However, the separate description of `model.py` assigns those responsibilities to `model.py`.

This documentation follows the explicit `model.py` description for the main data-flow diagram. Confirm the actual implementation and file names in the source code if both descriptions are intended to describe the same model wrapper.
