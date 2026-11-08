# MicroMind

**A tiny language model built from scratch — no dependencies, no
downloads, no frameworks. Just C.**

MicroMind is a character-level n-gram language model (order 3) with
Katz-style backoff smoothing and temperature sampling. You train it on
any text file, save the model, and generate text. ~230 lines of C.

## Try it

```bash
microai train corpus.txt        # learn from a text file -> model.bin
microai gen 500 "the quick"     # generate 500 chars seeded by "the quick"
microai gen 300 "art" 0.5       # lower temp = more predictable
microai repl                    # interactive: type a seed, get a continuation
microai save mymodel.bin        # save a trained model
microai load mymodel.bin        # load one
```

Windows binary `microai.exe` in Releases. Build:

```bash
gcc -O2 -o microai microai.exe microai.c -lm
```

## How it works

- **Vocabulary**: 34 chars (a–z, space, `.,!?'-` + newline)
- **Model**: trigram counts with bigram + unigram backoff —
  if a 3-char context was never seen, it falls back to 2 chars, then 1
- **Sampling**: roulette-wheel over `(count + 0.05)^(1/temperature)`
- **Size**: ~5.5 MB trained model (dense count tables)

## Honest expectations

This is a *statistical* language model — it learns which characters
follow which, not meaning. Output on the bundled 3 KB corpus is
recognizably word-like but drifts quickly. Train it on megabytes of
text and coherence improves dramatically. For real quality you'd want
a neural model — this exists to show the simplest possible "AI that
generates text" end to end: data → counts → sampling → output.

## Files

- `microai.c` — the whole model + CLI
- `corpus.txt` — bundled training text (original, public domain)
- `model.bin` — pre-trained model on corpus.txt (gitignored? no —
  regenerate with `microai train corpus.txt`)

MIT licensed.
