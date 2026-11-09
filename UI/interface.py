"""
interface.py  -  start here
================================

Chat with your decoder-only transformer.

    python interface.py                      # desktop window (Tkinter)
    python interface.py --cli                # terminal chat (works over SSH / without a display)
    python interface.py --checkpoint my.pt   # use a specific trained model
    python interface.py --train-steps 1500   # train on the built-in dialogues first, then chat
    python interface.py --train-steps 3000 --data my_chats.jsonl --size base

If `model.pt` exists in the current folder it is loaded automatically.
If no display is available (or Tkinter is missing) the terminal chat starts instead.

Files this app needs (keep them in one folder):
    transformer.py, model.py, engine.py, gui.py, interface.py
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import torch

from engine import (
    DEFAULT_SYSTEM,
    MODEL_PRESETS,
    ByteTokenizer,
    ChatEngine,
    ChatSession,
    GenerationSettings,
    TrainConfig,
    build_model,
    load_checkpoint,
    load_training_data,
    pick_device,
    save_checkpoint,
    train_model,
)

DEFAULT_CHECKPOINT = "trained_model.pt"

CLI_HELP = """\
Commands
  /help                     show this help
  /new                      start a new conversation
  /retry                    regenerate the last reply
  /system <text>            set the system prompt (starts a new conversation)
  /set <name> <value>       temp | top_k | top_p | max | rep | seed   (seed: a number or 'off')
  /settings                 show the current generation settings
  /model                    show model info
  /train [steps] [file]     train (continues the current model) on built-in dialogues or a file
  /save <file>              save the chat  (.json or .md)
  /load <file>              load a saved chat (.json)
  /ckpt <file>              save the model weights
  /quit                     exit            (Ctrl+C stops a reply that is being generated)
"""


# --------------------------------------------------------------------------- #
# Terminal chat
# --------------------------------------------------------------------------- #
def print_settings(s: GenerationSettings) -> None:
    print(
        f"  temperature={s.temperature}  top_k={s.top_k}  top_p={s.top_p}  "
        f"max_new_tokens={s.max_new_tokens}  repetition_penalty={s.repetition_penalty}  seed={s.seed}"
    )


def cli_train(engine: ChatEngine, session: ChatSession, steps: int, source: str, ckpt: str) -> None:
    data = load_training_data(source, engine.tok, engine.model.cfg, session.system_prompt)
    print(f"Training on {data.describe()} for {steps} steps (Ctrl+C to stop early)…")
    stop, start = threading.Event(), time.time()

    def progress(step: int, total: int, loss: float, lr: float) -> None:
        if step == 1 or step % 10 == 0 or step == total:
            rate = step / (time.time() - start)
            print(f"\r  step {step}/{total}  loss {loss:.3f}  {rate:.1f} steps/s  ETA {(total - step) / rate:,.0f}s   ", end="", flush=True)

    try:
        done = train_model(engine.model, data, TrainConfig(steps=steps), engine.device, progress, stop)
    except KeyboardInterrupt:
        stop.set()
        done = 0
        print("\nInterrupted.")
    engine.trained_steps += done
    engine.model.eval()
    save_checkpoint(engine.model, ckpt, engine.meta(session.system_prompt))
    print(f"\nDone. Saved model to {ckpt}")


def run_cli(engine: ChatEngine, session: ChatSession, settings: GenerationSettings, ckpt: str) -> None:
    print("\n" + "=" * 62)
    print(" TinyChat · terminal mode")
    print(" " + engine.describe())
    print(" Type a message, or /help for commands.  Ctrl+D or /quit to exit.")
    print("=" * 62)
    if not engine.is_trained:
        answer = input("\nThis model is untrained. Train it on the built-in dialogues now? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            cli_train(engine, session, 1500, "builtin", ckpt)

    names = {"temp": "temperature", "top_k": "top_k", "top_p": "top_p", "max": "max_new_tokens",
             "rep": "repetition_penalty", "seed": "seed"}

    def reply() -> None:
        stop = threading.Event()
        print("\033[1;36mtinychat>\033[0m ", end="", flush=True)
        parts = []
        try:
            for chunk in engine.reply_stream(session.messages, session.system_prompt, settings, stop):
                parts.append(chunk)
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            stop.set()
            print("  [stopped]", end="")
        text = "".join(parts).strip()
        if text:
            session.add("assistant", text)
        s = engine.last_stats
        print(f"\n\033[2m  ({s['tokens']} tokens, {s['tokens_per_second']:.0f} tok/s, "
              f"context {engine.context_tokens(session.messages, session.system_prompt)}/{engine.model.cfg.max_seq_len})\033[0m\n")

    while True:
        try:
            line = input("\033[1;32myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            session.add("user", line)
            reply()
            continue

        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        try:
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(CLI_HELP)
            elif cmd == "new":
                session.clear()
                print("  New conversation.\n")
            elif cmd == "retry":
                if session.messages and session.messages[-1].role == "assistant":
                    session.messages.pop()
                    reply()
                else:
                    print("  Nothing to retry.\n")
            elif cmd == "system":
                session.system_prompt = arg or DEFAULT_SYSTEM
                session.clear()
                print(f"  System prompt set. New conversation.\n")
            elif cmd == "set":
                key, _, value = arg.partition(" ")
                if key not in names:
                    print(f"  Unknown setting. Choose from: {', '.join(names)}\n")
                    continue
                attr = names[key]
                if attr == "seed":
                    settings.seed = None if value.strip() in ("off", "none", "") else int(value)
                elif attr in ("top_k", "max_new_tokens"):
                    setattr(settings, attr, int(value))
                else:
                    setattr(settings, attr, float(value))
                print_settings(settings)
                print()
            elif cmd == "settings":
                print_settings(settings)
                print()
            elif cmd == "model":
                print(" ", engine.describe(), f"· trained {engine.trained_steps} steps\n")
            elif cmd == "train":
                parts = arg.split(maxsplit=1)
                steps = int(parts[0]) if parts else 1500
                source = parts[1] if len(parts) > 1 else "builtin"
                cli_train(engine, session, steps, source, ckpt)
                print()
            elif cmd == "save":
                session.save(arg or "chat.json")
                print(f"  Saved chat to {arg or 'chat.json'}\n")
            elif cmd == "load":
                loaded = ChatSession.load(arg)
                session.system_prompt, session.messages = loaded.system_prompt, loaded.messages
                print(f"  Loaded {len(session.messages)} messages.\n")
            elif cmd == "ckpt":
                save_checkpoint(engine.model, arg or ckpt, engine.meta(session.system_prompt))
                print(f"  Saved model to {arg or ckpt}\n")
            else:
                print("  Unknown command. Type /help.\n")
        except Exception as exc:
            print(f"  Error: {type(exc).__name__}: {exc}\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with a decoder-only transformer.")
    p.add_argument("--cli", action="store_true", help="terminal chat instead of the desktop window")
    p.add_argument("--checkpoint", help=f"model file to load / save (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--size", choices=list(MODEL_PRESETS), default="small", help="size of a NEW model")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--train-steps", type=int, default=0, help="train this many steps before chatting")
    p.add_argument("--data", default="builtin", help="training data: 'builtin', a .jsonl/.json chat file or a .txt file")
    p.add_argument("--seed", type=int, help="random seed for weight init / training")
    p.add_argument("--theme", choices=["dark", "light"], default="dark")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    ckpt_path = args.checkpoint or DEFAULT_CHECKPOINT

    meta: dict = {}
    if Path(ckpt_path).is_file():
        model, meta = load_checkpoint(ckpt_path, device)
        print(f"Loaded model from {ckpt_path}")
    elif args.checkpoint:
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    else:
        model = build_model(args.size, device)
        print(f"Created a new '{args.size}' model (untrained).")

    engine = ChatEngine(model, ByteTokenizer(), device, trained_steps=int(meta.get("trained_steps", 0)))
    session = ChatSession(system_prompt=meta.get("system_prompt", DEFAULT_SYSTEM))
    settings = GenerationSettings()

    if args.train_steps > 0:
        cli_train(engine, session, args.train_steps, args.data, ckpt_path)

    if not args.cli:
        try:
            from gui import ChatApp

            ChatApp(engine, session, settings, checkpoint_path=ckpt_path, theme=args.theme).mainloop()
            return
        except ImportError:
            print("Tkinter is not installed - starting terminal chat instead.")
        except Exception as exc:  # e.g. TclError: no display
            if type(exc).__name__ != "TclError":
                raise
            print(f"No display available ({exc}) - starting terminal chat instead.")
    run_cli(engine, session, settings, ckpt_path)


if __name__ == "__main__":
    main()