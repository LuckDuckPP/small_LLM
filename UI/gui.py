"""
gui.py
===========

Desktop chat window (Tkinter - ships with Python, nothing extra to install).

Features
--------
* Streaming replies, token by token, generated in a background thread (UI never freezes)
* Stop button / Esc, Regenerate, Copy last reply
* Live generation controls: temperature, top-k, top-p, max tokens, repetition penalty, seed,
  one-click presets (Precise / Balanced / Creative)
* Editable system prompt
* Context meter that shows how much of the model's window the chat uses
  (oldest turns are dropped automatically when it is full)
* Train window: train from scratch or continue training, on the built-in sample dialogues,
  a chat .jsonl/.json file or a plain .txt file - with a live loss curve
* Load / save model checkpoints, load / save chats (JSON or Markdown)
* Light and dark theme

Shortcuts: Enter = send, Shift+Enter = new line, Esc = stop, Ctrl+N = new chat,
           Ctrl+S = save chat, Ctrl+R = regenerate, Ctrl+T = train window
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from engine import (
    DEFAULT_SYSTEM,
    MODEL_PRESETS,
    ChatEngine,
    ChatSession,
    GenerationSettings,
    TrainConfig,
    build_model,
    load_checkpoint,
    load_training_data,
    save_checkpoint,
    train_model,
)

APP_TITLE = "TinyChat · decoder-only transformer"
ASSISTANT_NAME = "TinyChat"

THEMES = {
    "light": dict(
        bg="#f3f4f6", panel="#ffffff", fg="#1f2328", muted="#6b7280", border="#d0d7de",
        user_bg="#dbeafe", bot_bg="#eceff3", accent="#2563eb", accent_hover="#1d4ed8",
        accent_fg="#ffffff", input_bg="#ffffff", note="#9a6700",
    ),
    "dark": dict(
        bg="#14161b", panel="#1b1e25", fg="#e6e8ec", muted="#98a2b3", border="#333947",
        user_bg="#1f3a5f", bot_bg="#272c37", accent="#3b82f6", accent_hover="#2563eb",
        accent_fg="#ffffff", input_bg="#232831", note="#e3b341",
    ),
}

PRESETS = {
    "Precise":  dict(temperature=0.3, top_k=20, top_p=0.8, rep=1.0),
    "Balanced": dict(temperature=0.7, top_k=40, top_p=0.9, rep=1.0),
    "Creative": dict(temperature=1.1, top_k=80, top_p=0.95, rep=1.05),
}


class LabeledScale(ttk.Frame):
    """A slider with a caption and a live value read-out."""

    def __init__(self, parent, text: str, var: tk.DoubleVar, lo: float, hi: float, fmt) -> None:
        super().__init__(parent)
        self.var, self.fmt = var, fmt
        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text=text).pack(side="left")
        self.value_label = ttk.Label(head, style="Muted.TLabel")
        self.value_label.pack(side="right")
        ttk.Scale(self, from_=lo, to=hi, variable=var).pack(fill="x", pady=(2, 0))
        var.trace_add("write", lambda *_: self._refresh())
        self._refresh()

    def _refresh(self) -> None:
        self.value_label.config(text=self.fmt(self.var.get()))


# =========================================================================== #
# Main window
# =========================================================================== #
class ChatApp(tk.Tk):
    def __init__(
        self,
        engine: ChatEngine,
        session: ChatSession,
        settings: GenerationSettings,
        checkpoint_path: str = "chat_model.pt",
        theme: str = "dark",
    ) -> None:
        super().__init__()
        self.engine, self.session = engine, session
        self.checkpoint_path = checkpoint_path
        self.theme = theme

        self.title(APP_TITLE)
        self.geometry("1200x780")
        self.minsize(960, 600)

        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.generating = False
        self.training_active = False
        self.partial = ""
        self.train_window: Optional["TrainWindow"] = None

        # generation settings bound to widgets
        self.var_temp = tk.DoubleVar(value=settings.temperature)
        self.var_topk = tk.DoubleVar(value=settings.top_k)
        self.var_topp = tk.DoubleVar(value=settings.top_p)
        self.var_max = tk.DoubleVar(value=settings.max_new_tokens)
        self.var_rep = tk.DoubleVar(value=settings.repetition_penalty)
        self.var_seed = tk.StringVar(value="" if settings.seed is None else str(settings.seed))
        self.var_status = tk.StringVar(value="Ready")
        self.var_theme_dark = tk.BooleanVar(value=(theme == "dark"))

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        base = tkfont.nametofont("TkDefaultFont")
        self.font_body = tkfont.Font(family=base.cget("family"), size=12)
        self.font_bold = tkfont.Font(family=base.cget("family"), size=10, weight="bold")
        self.font_small = tkfont.Font(family=base.cget("family"), size=9)

        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._configure_tags()
        self._apply_theme()
        self._bind_keys()

        self._render_session()
        self._refresh_model_info()
        self._show_untrained_notice()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(25, self._poll_events)
        self.input.focus_set()

    # ------------------------------------------------------------------ UI --- #
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(side="top", fill="x")
        self.toolbar_buttons = {}

        def add(key, text, cmd, side="left", style="TButton"):
            b = ttk.Button(bar, text=text, command=cmd, style=style)
            b.pack(side=side, padx=(0, 6) if side == "left" else (6, 0))
            self.toolbar_buttons[key] = b

        add("new", "＋ New chat", self.new_chat)
        add("save", "Save chat", self.save_chat)
        add("open", "Open chat", self.open_chat)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        add("regen", "↻ Regenerate", self.regenerate)
        add("copy", "Copy reply", self.copy_last_reply)
        add("train", "⚙ Train…", self.open_train_window, side="right", style="Accent.TButton")

    def _build_body(self) -> None:
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # ---- left: conversation + input ---- #
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.chat = tk.Text(
            left, wrap="word", state="disabled", relief="flat", borderwidth=0,
            padx=14, pady=12, font=self.font_body, cursor="arrow", highlightthickness=1,
        )
        scroll = ttk.Scrollbar(left, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        self.chat.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        entry = ttk.Frame(left)
        entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        entry.columnconfigure(0, weight=1)
        self.input = tk.Text(
            entry, height=3, wrap="word", relief="flat", padx=10, pady=8,
            font=self.font_body, highlightthickness=1, undo=True,
        )
        self.input.grid(row=0, column=0, sticky="ew")
        buttons = ttk.Frame(entry)
        buttons.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.send_btn = ttk.Button(buttons, text="Send  ⏎", style="Accent.TButton", command=self.send)
        self.send_btn.pack(fill="x")
        self.stop_btn = ttk.Button(buttons, text="■ Stop", command=self.stop_generation, state="disabled")
        self.stop_btn.pack(fill="x", pady=(6, 0))

        # ---- right: tabs ---- #
        side = ttk.Notebook(body, width=300)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        gen_tab = ttk.Frame(side, padding=12)
        model_tab = ttk.Frame(side, padding=12)
        side.add(gen_tab, text="Generation")
        side.add(model_tab, text="Model")
        self._build_generation_tab(gen_tab)
        self._build_model_tab(model_tab)

    def _build_generation_tab(self, tab: ttk.Frame) -> None:
        preset_row = ttk.Frame(tab)
        preset_row.pack(fill="x", pady=(0, 8))
        ttk.Label(preset_row, text="Preset").pack(side="left")
        for name in PRESETS:
            ttk.Button(preset_row, text=name, width=9, command=lambda n=name: self._apply_preset(n)).pack(
                side="left", padx=(6, 0)
            )

        pad = dict(fill="x", pady=5)
        LabeledScale(tab, "Temperature", self.var_temp, 0.0, 1.5, lambda v: f"{v:.2f}" if v > 0.01 else "greedy").pack(**pad)
        LabeledScale(tab, "Top-k", self.var_topk, 0, 200, lambda v: "off" if round(v) == 0 else str(round(v))).pack(**pad)
        LabeledScale(tab, "Top-p", self.var_topp, 0.1, 1.0, lambda v: "off" if v > 0.995 else f"{v:.2f}").pack(**pad)
        max_hi = max(32, self.engine.model.cfg.max_seq_len - 64)
        LabeledScale(tab, "Max new tokens", self.var_max, 16, max_hi, lambda v: str(round(v))).pack(**pad)
        LabeledScale(tab, "Repetition penalty", self.var_rep, 1.0, 1.5, lambda v: "off" if v < 1.005 else f"{v:.2f}").pack(**pad)

        seed_row = ttk.Frame(tab)
        seed_row.pack(**pad)
        ttk.Label(seed_row, text="Seed (blank = random)").pack(side="left")
        ttk.Entry(seed_row, textvariable=self.var_seed, width=9).pack(side="right")

        ttk.Separator(tab).pack(fill="x", pady=10)
        ttk.Label(tab, text="System prompt").pack(anchor="w")
        self.system_box = tk.Text(tab, height=5, wrap="word", relief="flat", padx=8, pady=6,
                                  font=self.font_small, highlightthickness=1)
        self.system_box.pack(fill="x", pady=(4, 6))
        self.system_box.insert("1.0", self.session.system_prompt)
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Button(row, text="Apply", command=self.apply_system_prompt).pack(side="left")
        ttk.Button(row, text="Reset", command=self.reset_system_prompt).pack(side="left", padx=6)

    def _build_model_tab(self, tab: ttk.Frame) -> None:
        self.model_info = ttk.Label(tab, wraplength=270, justify="left")
        self.model_info.pack(anchor="w")
        self.trained_info = ttk.Label(tab, wraplength=270, justify="left", style="Muted.TLabel")
        self.trained_info.pack(anchor="w", pady=(6, 10))
        ttk.Button(tab, text="Load model…", command=self.load_model).pack(fill="x", pady=3)
        ttk.Button(tab, text="Save model…", command=self.save_model).pack(fill="x", pady=3)
        ttk.Button(tab, text="Train…  (Ctrl+T)", command=self.open_train_window).pack(fill="x", pady=3)
        ttk.Separator(tab).pack(fill="x", pady=12)
        ttk.Checkbutton(tab, text="Dark theme", variable=self.var_theme_dark, command=self.toggle_theme).pack(anchor="w")
        ttk.Separator(tab).pack(fill="x", pady=12)
        ttk.Label(
            tab, style="Muted.TLabel", wraplength=270, justify="left",
            text=("Enter  send\nShift+Enter  new line\nEsc  stop\nCtrl+N  new chat\n"
                  "Ctrl+S  save chat\nCtrl+R  regenerate\nCtrl+T  train"),
        ).pack(anchor="w")

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, padding=(12, 4, 12, 8))
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.var_status, style="Muted.TLabel").pack(side="left")
        self.ctx_bar = ttk.Progressbar(bar, length=170, maximum=100)
        self.ctx_bar.pack(side="right")
        self.ctx_label = ttk.Label(bar, style="Muted.TLabel")
        self.ctx_label.pack(side="right", padx=(0, 8))

    # --------------------------------------------------------------- theme --- #
    def _configure_tags(self) -> None:
        t = self.chat
        t.tag_configure("user_name", font=self.font_bold, lmargin1=70, lmargin2=70, spacing1=4, justify="right")
        t.tag_configure("bot_name", font=self.font_bold, lmargin1=14, lmargin2=14, spacing1=4)
        t.tag_configure("user_body", lmargin1=70, lmargin2=70, rmargin=14, spacing1=6, spacing3=6)
        t.tag_configure("bot_body", lmargin1=14, lmargin2=14, rmargin=70, spacing1=6, spacing3=6)
        t.tag_configure("note", font=self.font_small, lmargin1=14, lmargin2=14, rmargin=14, spacing1=4, spacing3=4)
        t.tag_configure("spacer", font=("TkDefaultFont", 4))

    def _apply_theme(self) -> None:
        p, s = THEMES[self.theme], self.style
        self.configure(bg=p["bg"])
        s.configure(".", background=p["bg"], foreground=p["fg"], fieldbackground=p["input_bg"],
                    bordercolor=p["border"], lightcolor=p["border"], darkcolor=p["border"],
                    troughcolor=p["border"], focuscolor=p["bg"])
        s.configure("TFrame", background=p["bg"])
        s.configure("TLabel", background=p["bg"], foreground=p["fg"])
        s.configure("Muted.TLabel", foreground=p["muted"])
        s.configure("TButton", background=p["panel"], foreground=p["fg"], padding=(10, 5), borderwidth=1)
        s.map("TButton", background=[("active", p["border"]), ("disabled", p["bg"])],
              foreground=[("disabled", p["muted"])])
        s.configure("Accent.TButton", background=p["accent"], foreground=p["accent_fg"], bordercolor=p["accent"])
        s.map("Accent.TButton", background=[("active", p["accent_hover"]), ("disabled", p["border"])],
              foreground=[("disabled", p["muted"])])
        s.configure("TNotebook", background=p["bg"], borderwidth=0)
        s.configure("TNotebook.Tab", background=p["panel"], foreground=p["muted"], padding=(14, 6))
        s.map("TNotebook.Tab", background=[("selected", p["bg"])], foreground=[("selected", p["fg"])])
        s.configure("TCheckbutton", background=p["bg"], foreground=p["fg"])
        s.map("TCheckbutton", background=[("active", p["bg"])])
        s.configure("TScale", background=p["bg"], troughcolor=p["border"])
        s.configure("TEntry", fieldbackground=p["input_bg"], foreground=p["fg"], insertcolor=p["fg"])
        s.configure("TSeparator", background=p["border"])
        s.configure("Horizontal.TProgressbar", background=p["accent"], troughcolor=p["border"],
                    bordercolor=p["border"], lightcolor=p["accent"], darkcolor=p["accent"])
        s.configure("TCombobox", fieldbackground=p["input_bg"], foreground=p["fg"], background=p["panel"])

        for w in (self.chat, self.input, self.system_box):
            w.configure(bg=p["panel"] if w is self.chat else p["input_bg"], fg=p["fg"],
                        insertbackground=p["fg"], selectbackground=p["accent"], selectforeground="#ffffff",
                        highlightbackground=p["border"], highlightcolor=p["accent"])
        t = self.chat
        t.tag_configure("user_name", foreground=p["accent"])
        t.tag_configure("bot_name", foreground=p["muted"])
        t.tag_configure("user_body", background=p["user_bg"], foreground=p["fg"])
        t.tag_configure("bot_body", background=p["bot_bg"], foreground=p["fg"])
        t.tag_configure("note", foreground=p["note"])
        if self.train_window is not None and self.train_window.winfo_exists():
            self.train_window.apply_theme()

    def toggle_theme(self) -> None:
        self.theme = "dark" if self.var_theme_dark.get() else "light"
        self._apply_theme()

    # ------------------------------------------------------------ key binds --- #
    def _bind_keys(self) -> None:
        def bind(seq, fn):
            for w in (self, self.input, self.chat):
                w.bind(seq, lambda e, f=fn: (f(), "break")[1])

        bind("<Control-n>", self.new_chat)
        bind("<Control-s>", self.save_chat)
        bind("<Control-r>", self.regenerate)
        bind("<Control-t>", self.open_train_window)
        bind("<Escape>", self.stop_generation)
        self.input.bind("<Return>", lambda e: (self.send(), "break")[1])
        self.input.bind("<Shift-Return>", lambda e: (self.input.insert("insert", "\n"), "break")[1])

    # --------------------------------------------------------- chat display --- #
    def _insert(self, text: str, *tags: str) -> None:
        stick = self.chat.yview()[1] >= 0.999
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tags)
        self.chat.configure(state="disabled")
        if stick:
            self.chat.see("end")

    def _insert_header(self, role: str) -> None:
        if role == "user":
            self._insert("You\n", "user_name")
        else:
            self._insert(f"{ASSISTANT_NAME}\n", "bot_name")

    def _insert_message(self, role: str, text: str) -> None:
        self._insert_header(role)
        body = "user_body" if role == "user" else "bot_body"
        self._insert(text + "\n", body)
        self._insert("\n", "spacer")

    def _insert_note(self, text: str) -> None:
        self._insert(text + "\n", "note")
        self._insert("\n", "spacer")

    def _render_session(self) -> None:
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        if not self.session.messages:
            self._insert_note("Say hello to start chatting. Enter sends, Shift+Enter adds a new line.")
        for m in self.session.messages:
            self._insert_message(m.role, m.content)
        self._update_context()

    def _show_untrained_notice(self) -> None:
        if not self.engine.is_trained:
            self._insert_note(
                "⚠ This model has not been trained yet, so replies will be random bytes.\n"
                "Click “Train…” (Ctrl+T) and press Start to quick-train on the built-in sample dialogues, "
                "or load a trained model from the Model tab."
            )

    # ------------------------------------------------------------- settings --- #
    def _apply_preset(self, name: str) -> None:
        p = PRESETS[name]
        self.var_temp.set(p["temperature"])
        self.var_topk.set(p["top_k"])
        self.var_topp.set(p["top_p"])
        self.var_rep.set(p["rep"])

    def _collect_settings(self) -> GenerationSettings:
        seed_text = self.var_seed.get().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            seed = None
        return GenerationSettings(
            temperature=round(self.var_temp.get(), 2),
            top_k=int(round(self.var_topk.get())),
            top_p=round(self.var_topp.get(), 2),
            max_new_tokens=int(round(self.var_max.get())),
            repetition_penalty=round(self.var_rep.get(), 2),
            seed=seed,
        )

    def apply_system_prompt(self) -> None:
        self.session.system_prompt = self.system_box.get("1.0", "end").strip()
        self.var_status.set("System prompt updated")
        self._update_context()

    def reset_system_prompt(self) -> None:
        self.system_box.delete("1.0", "end")
        self.system_box.insert("1.0", DEFAULT_SYSTEM)
        self.apply_system_prompt()

    def _update_context(self) -> None:
        used = self.engine.context_tokens(self.session.messages, self.session.system_prompt)
        cap = self.engine.model.cfg.max_seq_len
        self.ctx_bar.configure(value=min(100, 100 * used / cap))
        self.ctx_label.config(text=f"Context {min(used, cap)} / {cap} tokens")

    def _refresh_model_info(self) -> None:
        self.model_info.config(text=self.engine.describe())
        if self.engine.is_trained:
            self.trained_info.config(text=f"Trained for {self.engine.trained_steps:,} steps.")
        else:
            self.trained_info.config(text="Not trained yet - open the Train window.")

    # ------------------------------------------------------------ generation --- #
    def _set_busy(self, busy: bool) -> None:
        self.send_btn.configure(state="disabled" if busy else "normal")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        for key in ("new", "regen"):
            self.toolbar_buttons[key].configure(state="disabled" if busy else "normal")

    def send(self) -> None:
        if self.generating:
            return
        if self.training_active:
            self.var_status.set("Training is running - chat is paused until it finishes.")
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        if not self.session.messages:
            self._render_session()      # clears the welcome note
            self._show_untrained_notice()
        self.session.add("user", text)
        self._insert_message("user", text)
        self._start_generation()

    def _start_generation(self) -> None:
        self.partial = ""
        self.stop_event.clear()
        self.generating = True
        self._set_busy(True)
        self.var_status.set("Generating…")
        self._insert_header("assistant")

        messages = list(self.session.messages)
        system_prompt = self.session.system_prompt
        settings = self._collect_settings()

        def work() -> None:
            try:
                for chunk in self.engine.reply_stream(messages, system_prompt, settings, self.stop_event):
                    self.events.put(("chunk", chunk))
                self.events.put(("done", dict(self.engine.last_stats)))
            except Exception as exc:  # surface errors in the UI instead of dying silently
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def stop_generation(self) -> None:
        if self.generating:
            self.stop_event.set()
            self.var_status.set("Stopping…")

    def regenerate(self) -> None:
        if self.generating or self.training_active:
            return
        if self.session.messages and self.session.messages[-1].role == "assistant":
            self.session.messages.pop()
            self._render_session()
            self._start_generation()

    def copy_last_reply(self) -> None:
        for m in reversed(self.session.messages):
            if m.role == "assistant":
                self.clipboard_clear()
                self.clipboard_append(m.content)
                self.var_status.set("Copied last reply to clipboard")
                return
        self.var_status.set("Nothing to copy yet")

    def _poll_events(self) -> None:
        chunks = []
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "chunk":
                    chunks.append(payload)
                    continue
                if chunks:
                    self._on_chunk("".join(chunks))
                    chunks = []
                if kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        if chunks:
            self._on_chunk("".join(chunks))
        self.after(25, self._poll_events)

    def _on_chunk(self, text: str) -> None:
        self.partial += text
        self._insert(text, "bot_body")

    def _finish_generation(self) -> None:
        self._insert("\n", "bot_body")
        self._insert("\n", "spacer")
        self.generating = False
        self._set_busy(False)
        self._update_context()

    def _on_done(self, stats: dict) -> None:
        reply = self.partial.strip()
        if reply:
            self.session.add("assistant", reply)
        stopped = self.stop_event.is_set()
        self._finish_generation()
        if not reply:
            self._insert_note("(empty reply - the model produced end-of-message immediately)")
        note = f"{stats['tokens']} tokens · {stats['tokens_per_second']:.1f} tok/s"
        if stats.get("dropped_messages"):
            note += f" · {stats['dropped_messages']} oldest message(s) no longer fit the context"
        self.var_status.set(("Stopped · " if stopped else "Done · ") + note)

    def _on_error(self, message: str) -> None:
        self._finish_generation()
        self._insert_note(f"Error: {message}")
        self.var_status.set("Error")

    # ----------------------------------------------------------- chat files --- #
    def new_chat(self) -> None:
        if self.generating:
            return
        self.session.clear()
        self._render_session()
        self._show_untrained_notice()
        self.var_status.set("New chat")

    def save_chat(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save chat", defaultextension=".json",
            filetypes=[("JSON chat", "*.json"), ("Markdown", "*.md")],
        )
        if path:
            self.session.save(path)
            self.var_status.set(f"Saved chat to {Path(path).name}")

    def open_chat(self) -> None:
        if self.generating:
            return
        path = filedialog.askopenfilename(title="Open chat", filetypes=[("JSON chat", "*.json")])
        if not path:
            return
        try:
            self.session = ChatSession.load(path)
        except Exception as exc:
            messagebox.showerror("Could not open chat", str(exc))
            return
        self.system_box.delete("1.0", "end")
        self.system_box.insert("1.0", self.session.system_prompt)
        self._render_session()
        self.var_status.set(f"Opened {Path(path).name}")

    # --------------------------------------------------------- model files --- #
    def load_model(self) -> None:
        if self.generating or self.training_active:
            return
        path = filedialog.askopenfilename(title="Load model", filetypes=[("PyTorch checkpoint", "*.pt *.pth")])
        if not path:
            return
        try:
            model, meta = load_checkpoint(path, self.engine.device)
        except Exception as exc:
            messagebox.showerror("Could not load model", str(exc))
            return
        self.engine.model = model
        self.engine.trained_steps = int(meta.get("trained_steps", 0))
        self.checkpoint_path = path
        self._refresh_model_info()
        self._update_context()
        self.var_status.set(f"Loaded {Path(path).name}")

    def save_model(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save model", defaultextension=".pt", initialfile=Path(self.checkpoint_path).name,
            filetypes=[("PyTorch checkpoint", "*.pt")],
        )
        if path:
            save_checkpoint(self.engine.model, path, self.engine.meta(self.session.system_prompt))
            self.checkpoint_path = path
            self.var_status.set(f"Saved model to {Path(path).name}")

    # ------------------------------------------------------------- training --- #
    def open_train_window(self) -> None:
        if self.generating:
            self.var_status.set("Wait for the reply to finish (or press Stop) before training.")
            return
        if self.train_window is not None and self.train_window.winfo_exists():
            self.train_window.lift()
            return
        self.train_window = TrainWindow(self)

    def on_training_finished(self) -> None:
        self.training_active = False
        self._refresh_model_info()
        self._update_context()
        self.var_status.set("Training finished - the model is ready to chat.")

    def _on_close(self) -> None:
        self.stop_event.set()
        if self.train_window is not None and self.train_window.winfo_exists():
            self.train_window.stop_event.set()
        self.destroy()


# =========================================================================== #
# Training window
# =========================================================================== #
class TrainWindow(tk.Toplevel):
    def __init__(self, app: ChatApp) -> None:
        super().__init__(app)
        self.app = app
        self.title("Train the model")
        self.geometry("560x600")
        self.resizable(False, False)
        self.transient(app)

        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False
        self.losses: list = []
        self.smooth: Optional[float] = None
        self.started_at = 0.0

        has_trained = app.engine.is_trained
        self.var_source = tk.StringVar(value="builtin")
        self.var_path = tk.StringVar(value="")
        self.var_scratch = tk.BooleanVar(value=not has_trained)
        self.var_size = tk.StringVar(value=self._current_size())
        self.var_steps = tk.StringVar(value="1500")
        self.var_batch = tk.StringVar(value="16")
        self.var_lr = tk.StringVar(value="0.002")
        self.var_autosave = tk.BooleanVar(value=True)
        self.var_msg = tk.StringVar(value="Pick a data source and press Start.")

        self._build()
        self.apply_theme()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

    def _current_size(self) -> str:
        d = self.app.engine.model.cfg.d_model
        for name, preset in MODEL_PRESETS.items():
            if preset["d_model"] == d:
                return name
        return "small"

    def _build(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        src = ttk.LabelFrame(root, text="  Training data  ", padding=10)
        src.pack(fill="x")
        ttk.Radiobutton(src, text="Built-in sample dialogues (50+ Q&A pairs about AI / Python)",
                        variable=self.var_source, value="builtin").pack(anchor="w")
        row = ttk.Frame(src)
        row.pack(fill="x", pady=(6, 0))
        ttk.Radiobutton(row, text="My file:", variable=self.var_source, value="file").pack(side="left")
        ttk.Entry(row, textvariable=self.var_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="left")
        ttk.Label(src, style="Muted.TLabel", wraplength=500, justify="left",
                  text=".jsonl / .json: chat data ({\"messages\":[{role,content}…]} or {\"prompt\",\"response\"})."
                       "  .txt: raw text.").pack(anchor="w", pady=(6, 0))

        opt = ttk.LabelFrame(root, text="  Settings  ", padding=10)
        opt.pack(fill="x", pady=10)
        grid = ttk.Frame(opt)
        grid.pack(fill="x")
        for i, (label, var) in enumerate((("Steps", self.var_steps), ("Batch size", self.var_batch), ("Learning rate", self.var_lr))):
            ttk.Label(grid, text=label).grid(row=0, column=2 * i, sticky="w", padx=(0, 4))
            ttk.Entry(grid, textvariable=var, width=8).grid(row=0, column=2 * i + 1, padx=(0, 14))
        row2 = ttk.Frame(opt)
        row2.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(row2, text="Start from scratch (new random weights)", variable=self.var_scratch).pack(side="left")
        ttk.Label(row2, text="  size").pack(side="left")
        ttk.Combobox(row2, textvariable=self.var_size, values=list(MODEL_PRESETS), width=7, state="readonly").pack(side="left", padx=4)
        ttk.Checkbutton(opt, text="Save checkpoint when finished", variable=self.var_autosave).pack(anchor="w", pady=(8, 0))

        btns = ttk.Frame(root)
        btns.pack(fill="x")
        self.start_btn = ttk.Button(btns, text="▶ Start training", style="Accent.TButton", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="■ Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        ttk.Button(btns, text="Close", command=self._on_close).pack(side="right")

        self.progress = ttk.Progressbar(root, maximum=100)
        self.progress.pack(fill="x", pady=(12, 4))
        ttk.Label(root, textvariable=self.var_msg, style="Muted.TLabel", wraplength=520).pack(anchor="w")
        self.canvas = tk.Canvas(root, height=190, highlightthickness=1)
        self.canvas.pack(fill="both", expand=True, pady=(8, 0))

    def apply_theme(self) -> None:
        p = THEMES[self.app.theme]
        self.configure(bg=p["bg"])
        self.canvas.configure(bg=p["panel"], highlightbackground=p["border"])
        self.app.style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
        self.app.style.configure("TLabelframe.Label", background=p["bg"], foreground=p["muted"])
        self.app.style.configure("TRadiobutton", background=p["bg"], foreground=p["fg"])
        self._draw_plot()

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Training data",
            filetypes=[("Text / chat data", "*.txt *.jsonl *.json"), ("All files", "*.*")],
        )
        if path:
            self.var_path.set(path)
            self.var_source.set("file")

    # ---------------------------------------------------------------- run --- #
    def start(self) -> None:
        if self.running:
            return
        try:
            steps, batch, lr = int(self.var_steps.get()), int(self.var_batch.get()), float(self.var_lr.get())
            assert steps > 0 and batch > 0 and lr > 0
        except (ValueError, AssertionError):
            messagebox.showerror("Invalid settings", "Steps, batch size and learning rate must be positive numbers.", parent=self)
            return
        source = self.var_path.get().strip() if self.var_source.get() == "file" else "builtin"
        if self.var_source.get() == "file" and not Path(source).is_file():
            messagebox.showerror("File not found", "Choose a training data file first.", parent=self)
            return

        self.running = True
        self.app.training_active = True
        self.stop_event.clear()
        self.losses, self.smooth, self.started_at = [], None, time.time()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.var_msg.set("Preparing data…")
        params = dict(steps=steps, batch=batch, lr=lr, source=source,
                      scratch=self.var_scratch.get(), size=self.var_size.get())
        threading.Thread(target=self._run, args=(params,), daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        self.var_msg.set("Stopping after the current step…")

    def _run(self, p: dict) -> None:
        eng = self.app.engine
        try:
            if p["scratch"]:
                model = build_model(p["size"], eng.device, max_seq_len=eng.model.cfg.max_seq_len)
            else:
                model = eng.model
            data = load_training_data(p["source"], eng.tok, model.cfg, self.app.session.system_prompt)
            self.events.put(("info", f"Data: {data.describe()} · model: {model.num_parameters() / 1e6:.2f}M parameters"))
            tc = TrainConfig(steps=p["steps"], batch_size=p["batch"], lr=p["lr"],
                             warmup_steps=min(50, max(1, p["steps"] // 10)))
            done = train_model(
                model, data, tc, eng.device,
                on_progress=lambda s, t, l, lr: self.events.put(("progress", (s, t, l))),
                stop_event=self.stop_event,
            )
            self.events.put(("done", (model, done, p["scratch"])))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _poll(self) -> None:
        if not self.winfo_exists():
            return
        latest, ended = None, False
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    latest = payload
                    step, total, loss = payload
                    self.smooth = loss if self.smooth is None else 0.9 * self.smooth + 0.1 * loss
                    self.losses.append(self.smooth)
                elif kind == "info":
                    self.var_msg.set(payload)
                elif kind == "done":
                    self._finish(*payload)
                    ended = True
                elif kind == "error":
                    self._fail(payload)
                    ended = True
        except queue.Empty:
            pass
        if latest is not None and not ended:
            step, total, loss = latest
            elapsed = time.time() - self.started_at
            eta = elapsed / step * (total - step)
            self.progress.configure(value=100 * step / total)
            self.var_msg.set(f"Step {step}/{total} · loss {self.smooth:.3f} · {step / elapsed:.1f} steps/s · ETA {eta:,.0f}s")
            self._draw_plot()
        self.after(120, self._poll)

    def _finish(self, model, steps_done: int, scratch: bool) -> None:
        eng = self.app.engine
        eng.model = model
        eng.trained_steps = steps_done if scratch else eng.trained_steps + steps_done
        if self.var_autosave.get() and steps_done > 0:
            save_checkpoint(model, self.app.checkpoint_path, eng.meta(self.app.session.system_prompt))
            saved = f" · saved to {Path(self.app.checkpoint_path).name}"
        else:
            saved = ""
        self.var_msg.set(f"Finished {steps_done} steps · final loss {self.smooth or 0:.3f}{saved}")
        self.progress.configure(value=100 if not self.stop_event.is_set() else self.progress["value"])
        self._reset_buttons()
        self.app.on_training_finished()
        self._draw_plot()

    def _fail(self, message: str) -> None:
        self.var_msg.set(f"Error: {message}")
        self._reset_buttons()
        self.app.training_active = False

    def _reset_buttons(self) -> None:
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _on_close(self) -> None:
        if self.running and not messagebox.askyesno("Training in progress", "Stop training and close?", parent=self):
            return
        self.stop_event.set()
        self.destroy()

    # --------------------------------------------------------------- plot --- #
    def _draw_plot(self) -> None:
        c = self.canvas
        c.delete("all")
        p = THEMES[self.app.theme]
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 100)
        if len(self.losses) < 2:
            c.create_text(w / 2, h / 2, text="loss curve appears here", fill=p["muted"])
            return
        data = self.losses[-600:]
        lo, hi = min(data), max(data)
        span = (hi - lo) or 1.0
        left, right, top, bottom = 44, w - 12, 14, h - 20
        for frac in (0, 0.5, 1):
            y = bottom - frac * (bottom - top)
            c.create_line(left, y, right, y, fill=p["border"])
            c.create_text(left - 6, y, text=f"{lo + frac * span:.2f}", anchor="e", fill=p["muted"], font=("TkDefaultFont", 8))
        pts = []
        for i, v in enumerate(data):
            x = left + (right - left) * i / (len(data) - 1)
            y = bottom - (v - lo) / span * (bottom - top)
            pts += [x, y]
        c.create_line(*pts, fill=p["accent"], width=2, smooth=True)
        c.create_text(right, h - 8, text=f"step {len(self.losses)}", anchor="e", fill=p["muted"], font=("TkDefaultFont", 8))