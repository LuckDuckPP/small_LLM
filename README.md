The project folder contain of two main folder in which the f0llowing happens :-

< the UI folder contain three files of which >
    >> engine.py works as binder of the decoder only transformer architecture of which it does the following :-
         
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
      It sits on top of "transformer.py" and "model.py".


   >> interface.py Chat with your decoder-only transformer :- 

   >> gui.py is the graphical design of the window that open after executing interface.py which is the file that should execute first it consist of :-
          * Streaming replies, token by token, generated in a background thread (UI never freezes)
          * Stop button / Esc, Regenerate, Copy last reply
          * Live generation controls: temperature, top-k, top-p, max tokens, repetition penalty, seed, one-click presets (Precise / Balanced / Creative)
           * Editable system prompt
          * Context meter that shows how much of the model's window the chat uses (oldest turns are dropped automatically when it is full)
           * Train window: train from scratch or continue training, on the built-in sample dialogues, a chat .jsonl/.json file or a plain .txt file - with a live loss curve
          * Load / save model checkpoints, load / save chats (JSON or Markdown)
          * Light and dark theme
>   
>
>
< the main folder consist of the actual architecture and model of the AI model >

  >> model.py is a Decoder-only language model for text generation how it operates :-
       <> It wires together:
                          token ids ──► Embedding ──► TransformerDecoder ──► LM head ──► logits over vocab
                                       (this file)    (transformer.py)                     (this file)

       <> uses the logits to do next-token prediction :-
            * training  : cross-entropy between logits[t] and the token that follows position t
             * inference : autoregressive sampling (temperature, top-k, top-p, repetition penalty, EOS handling) accelerated by a KV cache.


    >> transformer.py :-
       A modern, decoder-only Transformer stack (LLaMA-style) written in pure PyTorch.

            This file contains ONLY the transformer architecture (the "body" of the model).
            It does not contain the token embedding layer or the language-modelling head;
            those live in `language_model.py`, which wraps this decoder.
 
            Building blocks
            ---------------
            * TransformerConfig     - one dataclass holding every hyper-parameter
             * RMSNorm               - cheaper, more stable alternative to LayerNorm
            * RotaryEmbedding       - rotary positional embeddings (RoPE), cache-aware
             * KVCache               - pre-allocated key/value cache for fast generation
            * CausalSelfAttention   - grouped-query attention (GQA) + RoPE + KV cache, uses PyTorch's fused SDPA (FlashAttention) kernel
           * SwiGLUFeedForward     - gated feed-forward network (SwiGLU)
             * TransformerBlock      - pre-norm residual block (attention + FFN)
            * TransformerDecoder    - the stacked blocks + final norm

              Data flow of a single block (pre-norm):

                    x ──► RMSNorm ──► Attention(RoPE, GQA, causal) ──┐
                    │                                                 ▼
                    └─────────────────────────────────────────────── (+) ──► h
                    h ──► RMSNorm ──► SwiGLU FFN ───────────────────┐
                    │                                                ▼
                    └──────────────────────────────────────────── (+) ──► out
