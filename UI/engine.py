"""
engine.py
==============

Everything between "a trained transformer" and "a chat conversation", with no UI code:

  * ByteTokenizer      - dependency-free tokenizer (256 byte tokens + chat special tokens)
  * Chat template      - turns a conversation into token ids  (<|user|>...<|eos|><|assistant|>)
  * ChatSession        - message history, JSON / Markdown save + load
  * Datasets           - chat (JSONL) data with loss ONLY on assistant replies, or raw text
  * train_model        - AdamW + warm-up + cosine schedule, progress callback, stop flag
  * stream_tokens      - token-by-token sampling with KV cache (temperature / top-k / top-p /
                         repetition penalty), cancellable
  * ChatEngine         - builds the prompt (dropping old turns when the context is full) and
                         streams decoded text chunks
  * save/load_checkpoint, build_model, SAMPLE_PAIRS (built-in demo dialogues)

It sits on top of `transformer.py` and `model.py`.
"""

from __future__ import annotations

import codecs
import json
import math
import random
import threading
import sys
from pathlib import Path
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

main_dir = Path(__file__).resolve().parent.parent / "main"
sys.path.insert(0, str(main_dir))

from model import (
    DecoderOnlyLM,
    apply_repetition_penalty,
    top_k_filter,
    top_p_filter,
)
from transformer import TransformerConfig

DEFAULT_SYSTEM = "You are TinyChat, a small friendly assistant."


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]


class ByteTokenizer:
    """
    Byte-level tokenizer: ids 0-255 are raw UTF-8 bytes, 256+ are special tokens.

    Works for any language/emoji with no vocabulary file. Special tokens are added
    structurally by the chat template and are NEVER parsed out of user text, so a user
    cannot inject "<|assistant|>" into the conversation.
    """

    def __init__(self) -> None:
        self.special: Dict[str, int] = {t: 256 + i for i, t in enumerate(SPECIAL_TOKENS)}
        self.pad_id = self.special["<|pad|>"]
        self.bos_id = self.special["<|bos|>"]
        self.eos_id = self.special["<|eos|>"]
        self.system_id = self.special["<|system|>"]
        self.user_id = self.special["<|user|>"]
        self.assistant_id = self.special["<|assistant|>"]
        self.vocab_size = 256 + len(SPECIAL_TOKENS)
        self.stop_ids = set(self.special.values())  # any special token ends a reply

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def incremental_decoder(self):
        """Decoder that buffers partial multi-byte characters while streaming."""
        return codecs.getincrementaldecoder("utf-8")(errors="replace")

    def role_id(self, role: str) -> int:
        return {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}[role]


# --------------------------------------------------------------------------- #
# Conversation + chat template
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str      # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatSession:
    system_prompt: str = DEFAULT_SYSTEM
    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))

    def clear(self) -> None:
        self.messages.clear()

    # ---- persistence ---- #
    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChatSession":
        msgs = [Message(m["role"], m["content"]) for m in d.get("messages", [])]
        return cls(system_prompt=d.get("system_prompt", DEFAULT_SYSTEM), messages=msgs)

    def to_markdown(self, assistant_name: str = "TinyChat") -> str:
        lines = [f"# Chat with {assistant_name}", "", f"*System prompt:* {self.system_prompt}", ""]
        for m in self.messages:
            who = "You" if m.role == "user" else assistant_name
            lines += [f"**{who}:** {m.content}", ""]
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.suffix.lower() == ".md":
            path.write_text(self.to_markdown(), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ChatSession":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def encode_message(tok: ByteTokenizer, role: str, content: str) -> List[int]:
    """<|role|> + utf-8 bytes + <|eos|>"""
    return [tok.role_id(role)] + tok.encode(content) + [tok.eos_id]


def build_prompt(
    tok: ByteTokenizer,
    messages: Sequence[Message],
    system_prompt: str,
    budget: int,
) -> Tuple[List[int], int]:
    """
    Build the model input for the next assistant reply:

        <|bos|> <|system|>...<|eos|> <|user|>...<|eos|> <|assistant|>...<|eos|> ... <|assistant|>

    If the history does not fit into `budget` tokens, the OLDEST turns are dropped
    (the system prompt is always kept). Returns (token_ids, number_of_dropped_messages).
    """
    sys_ids = tok.encode(system_prompt)[: max(0, budget // 3)] if system_prompt else []
    head = [tok.bos_id] + ([tok.system_id] + sys_ids + [tok.eos_id] if system_prompt else [])
    tail = [tok.assistant_id]
    available = max(budget - len(head) - len(tail), 2)

    kept: List[List[int]] = []
    used = 0
    for msg in reversed(messages):
        seg = encode_message(tok, msg.role, msg.content)
        if used + len(seg) > available:
            if not kept:  # even the newest message is too long: keep its end
                kept.append([seg[0]] + seg[-(available - 1):])
            break
        kept.append(seg)
        used += len(seg)
    kept.reverse()
    dropped = len(messages) - len(kept)
    return head + [i for seg in kept for i in seg] + tail, dropped


# --------------------------------------------------------------------------- #
# Built-in demo dialogues (so the app can be trained with no data file)
# --------------------------------------------------------------------------- #
SAMPLE_PAIRS: List[Tuple[str, str]] = [
    ("hi", "Hello! How can I help you today?"),
    ("hello", "Hi there! What would you like to talk about?"),
    ("hey", "Hey! Nice to meet you. Ask me anything."),
    ("good morning", "Good morning! I hope your day is going well."),
    ("good night", "Good night! Sleep well and see you soon."),
    ("how are you?", "I'm doing well, thank you for asking! How are you?"),
    ("what is your name?", "I'm TinyChat, a small decoder-only transformer running on your computer."),
    ("who made you?", "I was built from scratch in Python with PyTorch: embeddings, attention and a language model head."),
    ("what can you do?", "I can chat, answer simple questions and explain basic ideas about Python and transformers."),
    ("what is a transformer?", "A transformer is a neural network that uses attention to relate every token in a sequence to the tokens before it."),
    ("what is attention?", "Attention lets each token look at other tokens and decide which ones matter most for predicting what comes next."),
    ("what is a token?", "A token is a small piece of text. I read raw bytes, so each token here is a single byte."),
    ("how do you generate text?", "I predict one token at a time. Each new token is sampled from a probability distribution and fed back in."),
    ("what is temperature?", "Temperature controls randomness. Low values make me predictable, high values make me more creative."),
    ("what is top-k sampling?", "Top-k sampling keeps only the k most likely next tokens and samples from them."),
    ("what is top-p sampling?", "Top-p sampling keeps the smallest set of tokens whose probabilities add up to p."),
    ("what is a kv cache?", "A KV cache stores past keys and values so I do not have to recompute them for every new token."),
    ("what is rope?", "RoPE rotates query and key vectors by an angle that depends on position, which encodes relative distance."),
    ("what is python?", "Python is a popular programming language known for clear, readable code."),
    ("how do i print in python?", "Use the print function, for example: print('Hello, world!')"),
    ("how do i make a list in python?", "Put values in square brackets, for example: numbers = [1, 2, 3]"),
    ("how do i write a loop in python?", "Use for, for example: for i in range(5): print(i)"),
    ("what is pytorch?", "PyTorch is a Python library for building and training neural networks."),
    ("what is machine learning?", "Machine learning is when a computer learns patterns from data instead of following hand-written rules."),
    ("what is a neural network?", "A neural network is a stack of layers of simple math units that learn from examples."),
    ("what is training?", "Training means adjusting the model's weights so it predicts the next token better. The loss measures how wrong it is."),
    ("what is the loss?", "The loss is a number that says how far my predictions are from the right answer. Lower is better."),
    ("what is the capital of india?", "The capital of India is New Delhi."),
    ("what is the capital of france?", "The capital of France is Paris."),
    ("what is the capital of japan?", "The capital of Japan is Tokyo."),
    ("what is 2 plus 2?", "2 plus 2 is 4."),
    ("what is 10 plus 5?", "10 plus 5 is 15."),
    ("what is 3 times 4?", "3 times 4 is 12."),
    ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
    ("tell me another joke", "Why did the neural network go to school? To improve its learning rate!"),
    ("thanks", "You're welcome! Happy to help."),
    ("thank you", "My pleasure! Let me know if you need anything else."),
    ("bye", "Goodbye! Have a great day."),
    ("i am sad", "I'm sorry to hear that. Take a deep breath, and maybe talk to someone you trust."),
    ("i am bored", "Let's fix that! Ask me a question or I can tell you a joke."),
    ("can you help me learn python?", "Of course! Start with variables, loops and functions, then practice by building small programs."),
    ("are you a human?", "No, I'm a small language model. I only know patterns I learned from my training text."),
    ("do you make mistakes?", "Yes, often. I'm tiny, so please double-check anything important."),
    ("how big are you?", "I have only a few million parameters, which is tiny compared to large language models."),
    ("what is a language model?", "A language model predicts the next token given the tokens before it."),
    ("what is a gpu?", "A GPU is a chip that does many calculations in parallel, which makes neural network training much faster."),
    ("what is an embedding?", "An embedding turns a token id into a vector of numbers that the network can work with."),
    ("what is softmax?", "Softmax turns raw scores into probabilities that add up to one."),
    ("what is overfitting?", "Overfitting is when a model memorizes its training data instead of learning general patterns."),
    ("what is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("how many days are in a week?", "There are seven days in a week."),
    ("what color is the sky?", "On a clear day the sky looks blue because air scatters blue light the most."),
]


def sample_conversations(n_multi_turn: int = 300, seed: int = 0) -> List[List[Message]]:
    """Every pair as a single-turn chat, plus random 2-3 turn chats to teach multi-turn format."""
    rng = random.Random(seed)
    convs = [[Message("user", q), Message("assistant", a)] for q, a in SAMPLE_PAIRS] * 3
    for _ in range(n_multi_turn):
        turns = rng.sample(SAMPLE_PAIRS, k=rng.choice([2, 2, 3]))
        conv: List[Message] = []
        for q, a in turns:
            conv += [Message("user", q), Message("assistant", a)]
        convs.append(conv)
    return convs


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class ChatDataset:
    """
    Chat conversations encoded with the template above. The loss is computed ONLY on the
    assistant's words (and its closing <|eos|>); system/user tokens are masked with -100,
    so the model learns to *answer*, not to imitate the user.

    Sequences are right-padded per batch. With causal attention, right padding cannot
    influence earlier positions, so no padding mask is required.
    """

    def __init__(
        self,
        tok: ByteTokenizer,
        conversations: Sequence[Sequence[Message]],
        max_len: int,
        system_prompt: str = DEFAULT_SYSTEM,
    ) -> None:
        self.tok = tok
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for conv in conversations:
            ids = [tok.bos_id]
            learn = [False]
            if system_prompt:
                seg = encode_message(tok, "system", system_prompt)
                ids += seg
                learn += [False] * len(seg)
            for m in conv:
                seg = encode_message(tok, m.role, m.content)
                ids += seg
                learn += [False] + [m.role == "assistant"] * (len(seg) - 1)
            ids, learn = ids[: max_len + 1], learn[: max_len + 1]
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(
                [t if flag else -100 for t, flag in zip(ids[1:], learn[1:])], dtype=torch.long
            )
            if (y != -100).any():
                self.examples.append((x, y))
        if not self.examples:
            raise ValueError("No usable conversations (need at least one assistant message).")

    def __len__(self) -> int:
        return len(self.examples)

    def describe(self) -> str:
        n_tok = sum(len(x) for x, _ in self.examples)
        return f"{len(self.examples)} conversations, {n_tok:,} tokens"

    def batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        chosen = random.choices(self.examples, k=batch_size)
        length = max(len(x) for x, _ in chosen)
        xb = torch.full((batch_size, length), self.tok.pad_id, dtype=torch.long)
        yb = torch.full((batch_size, length), -100, dtype=torch.long)
        for i, (x, y) in enumerate(chosen):
            xb[i, : len(x)] = x
            yb[i, : len(y)] = y
        return xb.to(device), yb.to(device)


class TextDataset:
    """Raw text (plain language modelling): random windows, loss on every token."""

    def __init__(self, tok: ByteTokenizer, text: str, block_size: int) -> None:
        ids = tok.encode(text)
        while len(ids) < block_size + 2:      # tiny files: repeat so a window fits
            ids = ids + ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block = block_size

    def __len__(self) -> int:
        return len(self.data)

    def describe(self) -> str:
        return f"raw text, {len(self.data):,} bytes"

    def batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(self.data) - self.block - 1, (batch_size,))
        x = torch.stack([self.data[s : s + self.block] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.block + 1] for s in starts])
        return x.to(device), y.to(device)


def _parse_conversation(obj: dict) -> Optional[List[Message]]:
    if "messages" in obj:
        return [Message(m["role"], m["content"]) for m in obj["messages"]]
    for q_key, a_key in (("prompt", "response"), ("user", "assistant"), ("instruction", "output")):
        if q_key in obj and a_key in obj:
            return [Message("user", str(obj[q_key])), Message("assistant", str(obj[a_key]))]
    return None


def load_training_data(
    source: Optional[str],
    tok: ByteTokenizer,
    cfg: TransformerConfig,
    system_prompt: str = DEFAULT_SYSTEM,
):
    """
    source = None / "builtin" -> bundled sample dialogues
    *.jsonl  -> one JSON object per line: {"messages":[{"role","content"},...]}
                 or {"prompt": ..., "response": ...} (also user/assistant, instruction/output)
    *.json   -> a JSON list of the same objects
    anything else (*.txt ...) -> raw text language modelling
    """
    if not source or source == "builtin":
        return ChatDataset(tok, sample_conversations(), cfg.max_seq_len, system_prompt)

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        raw = path.read_text(encoding="utf-8")
        objs = (
            [json.loads(line) for line in raw.splitlines() if line.strip()]
            if suffix == ".jsonl"
            else json.loads(raw)
        )
        convs = [c for c in (_parse_conversation(o) for o in objs) if c]
        if not convs:
            raise ValueError(f"No conversations found in {path.name}")
        return ChatDataset(tok, convs, cfg.max_seq_len, system_prompt)
    return TextDataset(tok, path.read_text(encoding="utf-8", errors="replace"), min(256, cfg.max_seq_len))


# --------------------------------------------------------------------------- #
# Model factory + checkpoints
# --------------------------------------------------------------------------- #
MODEL_PRESETS = {
    "tiny":  dict(d_model=128, n_layers=3, n_heads=4, n_kv_heads=2),   # ~0.7M params
    "small": dict(d_model=192, n_layers=4, n_heads=6, n_kv_heads=3),   # ~2.1M params
    "base":  dict(d_model=256, n_layers=6, n_heads=8, n_kv_heads=4),   # ~5.9M params
}


def pick_device(name: Optional[str] = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    size: str = "small",
    device: Union[str, torch.device] = "cpu",
    max_seq_len: int = 512,
    dropout: float = 0.0,
) -> DecoderOnlyLM:
    tok = ByteTokenizer()
    cfg = TransformerConfig(
        vocab_size=tok.vocab_size, max_seq_len=max_seq_len, dropout=dropout, **MODEL_PRESETS[size]
    )
    return DecoderOnlyLM(cfg).to(device)


def save_checkpoint(model: DecoderOnlyLM, path: Union[str, Path], meta: Optional[dict] = None) -> None:
    torch.save(
        {
            "config": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "tokenizer": "byte-v1",
            "meta": meta or {},
        },
        str(path),
    )


def load_checkpoint(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    ckpt = torch.load(str(path), map_location=device)
    cfg = TransformerConfig.from_dict(ckpt["config"])
    if cfg.vocab_size != ByteTokenizer().vocab_size:
        raise ValueError(
            f"Checkpoint vocab size is {cfg.vocab_size}, but this chat app uses the byte "
            f"tokenizer ({ByteTokenizer().vocab_size}). It was trained with a different tokenizer."
        )
    model = DecoderOnlyLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt.get("meta", {})


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 16
    lr: float = 2e-3
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0


def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warm-up, then cosine decay to min_lr_ratio * lr."""
    if step <= tc.warmup_steps:
        return tc.lr * step / max(1, tc.warmup_steps)
    progress = (step - tc.warmup_steps) / max(1, tc.steps - tc.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return tc.lr * (tc.min_lr_ratio + (1 - tc.min_lr_ratio) * cosine)


def train_model(
    model: DecoderOnlyLM,
    dataset,
    tc: TrainConfig,
    device: Union[str, torch.device] = "cpu",
    on_progress: Optional[Callable[[int, int, float, float], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> int:
    """Train in place. Returns the number of optimizer steps actually performed."""
    device = torch.device(device)
    model.to(device).train()
    optimizer = model.configure_optimizers(lr=tc.lr, weight_decay=tc.weight_decay)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    done = 0
    try:
        for step in range(1, tc.steps + 1):
            if stop_event is not None and stop_event.is_set():
                break
            lr = lr_at(step, tc)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x, y = dataset.batch(tc.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(x, targets=y).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            done = step
            if on_progress is not None:
                on_progress(step, tc.steps, float(loss.item()), lr)
    finally:
        model.eval()
    return done


# --------------------------------------------------------------------------- #
# Streaming generation
# --------------------------------------------------------------------------- #
@dataclass
class GenerationSettings:
    temperature: float = 0.7          # 0 = greedy
    top_k: int = 40                   # 0 = off
    top_p: float = 0.9                # 1.0 = off
    max_new_tokens: int = 160
    repetition_penalty: float = 1.0   # 1.0 = off (byte-level: keep low, letters repeat a lot)
    repetition_window: int = 64       # only the last N tokens are penalised
    seed: Optional[int] = None


@torch.no_grad()
def stream_tokens(
    model: DecoderOnlyLM,
    prompt_ids: Sequence[int],
    settings: GenerationSettings,
    stop_ids: set,
    stop_event: Optional[threading.Event] = None,
) -> Iterator[int]:
    """Yield one sampled token id at a time (prefill once, then 1-token steps with the KV cache)."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        max_len = model.cfg.max_seq_len
        max_new = max(1, min(settings.max_new_tokens, max_len - len(prompt_ids)))

        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(settings.seed))

        cache = model.new_kv_cache(1)
        logits = model(ids, kv_cache=cache, last_token_only=True).logits[:, -1]   # prefill
        history = ids

        for step in range(max_new):
            if stop_event is not None and stop_event.is_set():
                return
            scores = logits.float()
            if settings.repetition_penalty != 1.0:
                window = history[:, -max(1, settings.repetition_window):]
                scores = apply_repetition_penalty(scores, window, settings.repetition_penalty)

            if settings.temperature <= 0:
                nxt = scores.argmax(dim=-1)
            else:
                scores = scores / settings.temperature
                if settings.top_k > 0:
                    scores = top_k_filter(scores, settings.top_k)
                if settings.top_p < 1.0:
                    scores = top_p_filter(scores, settings.top_p)
                probs = F.softmax(scores, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

            token = int(nxt.item())
            if token in stop_ids:
                return
            yield token
            if step == max_new - 1:
                return
            history = torch.cat([history, nxt[:, None]], dim=1)
            logits = model(nxt[:, None], kv_cache=cache, last_token_only=True).logits[:, -1]
    finally:
        model.train(was_training)


class ChatEngine:
    """Model + tokenizer + the logic that turns a conversation into a streamed reply."""

    def __init__(
        self,
        model: DecoderOnlyLM,
        tokenizer: Optional[ByteTokenizer] = None,
        device: Union[str, torch.device] = "cpu",
        trained_steps: int = 0,
    ) -> None:
        self.model = model.to(device).eval()
        self.tok = tokenizer or ByteTokenizer()
        self.device = torch.device(device)
        self.trained_steps = trained_steps
        self.last_stats: Dict[str, float] = {}

    # ---- info ---- #
    def describe(self) -> str:
        c = self.model.cfg
        return (
            f"{self.model.num_parameters() / 1e6:.2f}M parameters · {c.n_layers} layers · "
            f"d_model {c.d_model} · {c.n_heads}/{c.n_kv_heads} heads (GQA) · "
            f"context {c.max_seq_len} · {self.device.type.upper()}"
        )

    @property
    def is_trained(self) -> bool:
        return self.trained_steps > 0

    def context_tokens(self, messages: Sequence[Message], system_prompt: str) -> int:
        ids, _ = build_prompt(self.tok, messages, system_prompt, budget=10**9)
        return len(ids)

    def meta(self, system_prompt: str = DEFAULT_SYSTEM) -> dict:
        return {"trained_steps": self.trained_steps, "system_prompt": system_prompt}

    # ---- generation ---- #
    def reply_stream(
        self,
        messages: Sequence[Message],
        system_prompt: str,
        settings: GenerationSettings,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield decoded text chunks of the assistant's reply as they are generated."""
        max_len = self.model.cfg.max_seq_len
        max_new = max(1, min(settings.max_new_tokens, max_len - 16))
        prompt, dropped = build_prompt(self.tok, messages, system_prompt, budget=max_len - max_new)

        decoder = self.tok.incremental_decoder()
        n_tokens, start = 0, time.perf_counter()
        try:
            for token in stream_tokens(self.model, prompt, settings, self.tok.stop_ids, stop_event):
                n_tokens += 1
                text = decoder.decode(bytes([token]))
                if text:
                    yield text
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail
        finally:
            elapsed = max(time.perf_counter() - start, 1e-6)
            self.last_stats = {
                "tokens": n_tokens,
                "seconds": elapsed,
                "tokens_per_second": n_tokens / elapsed,
                "prompt_tokens": len(prompt),
                "dropped_messages": dropped,
            }