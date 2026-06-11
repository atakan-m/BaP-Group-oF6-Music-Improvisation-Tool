# Jazz Improvisation Tool — Code Reference

A complete reference for every module, class, and function in the codebase. Written for the maintainer who needs to find what something does without re-reading the source.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Repository layout](#2-repository-layout)
3. [End-to-end pipeline](#3-end-to-end-pipeline)
4. [Conventions: tokens, ticks, chords, columns](#4-conventions-tokens-ticks-chords-columns)
5. [ML/ — machine-learning stack](#5-ml--machine-learning-stack)
   - 5.1 [`chord_utils.py`](#51-chord_utilspy)
   - 5.2 [`model.py`](#52-modelpy)
   - 5.3 [`data_prep.py`](#53-data_preppy)
   - 5.4 [`train.py`](#54-trainpy)
   - 5.5 [`engine.py`](#55-enginepy) ← the realtime runtime
   - 5.6 [`generate.py`](#56-generatepy)
   - 5.7 [`perf.py`](#57-perfpy)
6. [SP/ — signal processing](#6-sp--signal-processing)
   - 6.1 [`signalprocessing.py`](#61-signalprocessingpy)
7. [HW/bap/ — hardware-side inputs](#7-hwbap--hardware-side-inputs)
   - 7.1 [`Buttons_v2.py`](#71-buttons_v2py)
   - 7.2 [`bapv1.py`](#72-bapv1py-deprecated)
8. [`Piano_display.py` — main application](#8-piano_displaypy--main-application)
9. [Operational notes](#9-operational-notes)
10. [Parked decisions / known artifacts](#10-parked-decisions--known-artifacts)

---

## 1. What this project is

A realtime jazz-improvisation tool. The player provides a chord progression and plays a microphone-tracked instrument; an LSTM model trained on the Weimar Jazz Database generates a 16th-note-resolution monophonic melody that scrolls down a pygame display. The model adapts to the player in realtime: if the player plays the note the model expected (or a "tolerable" near-miss), the rollout shifts forward by one tick; if the player attacks a different pitch, the engine declares a *deviation*, rebuilds its plan around the deviation note, and the display visibly re-paints the upcoming melody.

The intended deployment target is a Raspberry Pi 4B with a USB microphone.

---

## 2. Repository layout

```
BaP-Group-oF6-Music-Improvisation-Tool/
├── Piano_display.py             ← main entry point (pygame game loop)
├── ML/                          ← model, training, realtime engine
│   ├── engine.py                ← JazzImprov realtime engine + MicTickAggregator
│   ├── model.py                 ← JazzLSTM next-token predictor
│   ├── chord_utils.py           ← chord parsing + vocab lookups
│   ├── data_prep.py             ← raw DB → training tensors
│   ├── train.py                 ← training loop
│   ├── generate.py              ← offline MIDI render CLI + re-exports
│   ├── perf.py                  ← PerfLogger for timing measurements
│   ├── data/
│   │   ├── raw/wjazzd.db        ← Weimar Jazz Database (input)
│   │   └── processed/           ← grid_solos.parquet + chord_vocab.json + …
│   └── models/                  ← trained .pt checkpoints
├── SP/
│   └── signalprocessing.py      ← realtime pitch detector (HPS + onset)
├── HW/
│   └── bap/
│       ├── Buttons_v2.py        ← chord-progression entry UI
│       └── bapv1.py             ← old detector (kept for legacy refs)
└── CODE_REFERENCE.md            ← this file
```

---

## 3. End-to-end pipeline

```
                ┌───────────────────────────────────────────────────────┐
                │  OFFLINE (one-time)                                   │
                │                                                       │
                │  Weimar DB     ─►  data_prep.main()                   │
                │                    • parse beats / notes              │
                │                    • forward-fill chords              │
                │                    • transpose every solo to C tonic  │
                │                    • quantise to 16th-note grid       │
                │                                                       │
                │  → grid_solos.parquet + chord_vocab.json              │
                │                                                       │
                │  train.main()      • JazzLSTM next-token prediction   │
                │                    • cross-entropy, Adam, cosine LR   │
                │                                                       │
                │  → ML/models/jazz_lstm*.pt                            │
                └───────────────────────────────────────────────────────┘

                ┌───────────────────────────────────────────────────────┐
                │  REALTIME (Piano_display.py)                          │
                │                                                       │
                │  Buttons_v2.main()  ─►  chord progression (list)      │
                │  signalprocessing.PitchDetector  → detector.last_midi │
                │                                                       │
                │       │                       │                       │
                │       ▼                       ▼                       │
                │  make_realtime_engine    sounddevice.InputStream      │
                │    (loads .pt,           callback per ~11.6 ms        │
                │     int8-quantises)      → mic_aggregator.observe(..) │
                │       │                       │                       │
                │       ▼                       ▼                       │
                │   engine                 MicTickAggregator            │
                │       │                       │                       │
                │       └─── per 16th tick ─────┘                       │
                │                                                       │
                │       Main loop: aggregator → engine.commit(token)    │
                │                            → rollout → display update │
                │                                                       │
                │  → pygame screen                                      │
                └───────────────────────────────────────────────────────┘
```

The realtime loop is driven by pygame's frame clock (60 FPS). Every `Testing_variable_for_testing` frames (= one 16th note at the configured BPM) is one *tick*. At each tick: read the mic-aggregator's consensus token, commit it to the engine, repaint on-screen figures from the new rollout, spawn the next figure for the top of the screen.

---

## 4. Conventions: tokens, ticks, chords, columns

### Tokens

The model's vocabulary covers 59 tokens (see `model.py`):

| ID         | Meaning                                  |
|------------|------------------------------------------|
| 0          | `TOK_REST`                               |
| 1          | `TOK_HOLD` (sustain the previous pitch)  |
| 2 .. 58    | `NOTE_<midi>` for MIDI in `[40, 96]`     |

So `NOTE_60` (middle C) is token ID `2 + (60 − 40) = 22`. The base is at `TOK_NOTE_BASE = 2` and the MIDI offset is `PITCH_LOW = 40`.

### Ticks

- 1 tick = one 16th note (`TICKS_PER_BEAT = 4`, `TICKS_PER_BAR = 16`).
- Engine-side ticks: `engine.current_tick` increments by 1 on every `commit()`.
- Wall-clock ticks: `tally` increments by 1 in `Piano_display.py` for each 16th-note boundary.
- The two are offset by 31: with the 16-row pre-roll buffer, engine tick K reaches the play line at wall-clock tally `K + 31`. This is encoded in `_PLAY_LINE_OFFSET = 31`.

### Chords

Each chord is `(root_pc, normalized_quality)`:

- `root_pc` is the pitch class 0..11 (C, C#, D, …) or 12 for unknown.
- `normalized_quality` is a short string canonicalised from the source ("maj7" → "j7", "min7" → "-7", …).

The model embeds root and quality separately, so it shares knowledge across all chords with the same quality (every "m7" behaves similarly relative to its root).

All training data is **transposed so each solo's tonic is C**. There is no inverse transposition at inference time — that's a known limitation (see §10).

### Columns (display)

The visible piano spans 36 semitones from MIDI 48 (C3, column 0) to MIDI 83 (B5, column 35). `midi_to_col(midi) = midi − 48`, returning `None` for out-of-range pitches.

---

## 5. ML/ — machine-learning stack

### 5.1 `chord_utils.py`

Shared chord parsing. Used by `data_prep.py`, `engine.py`, and `generate.py`.

**Module constants**

| Symbol         | Value                                            |
|----------------|--------------------------------------------------|
| `NOTE_TO_PC`   | `{"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}` — letter → natural pitch class |
| `PITCH_NAMES`  | 12-element list of sharp-spelled note names      |
| `UNK_ROOT_ID`  | `12` — reserved root id for unparseable chords   |

**Functions**

#### `parse_root(token: str) → (int, str) | None`

Splits a chord head into `(root_pc, remainder)`. Handles `#`/`b` accidentals after the letter. Returns `None` if the leading character isn't a valid note letter.

- *Input*: a chord string like `"Bb7"` or `"C#m"`.
- *Output*: `(10, "7")` or `(1, "m")` respectively, or `None`.

#### `normalize_quality(q: str) → str`

Canonicalises chord-quality fragments. Maps "maj7"/"Maj7"/"M7" → "j7", "min7"/"Min7" → "-7", "min"/"Min" → "-", "dim" → "o", "aug" → "+", standalone "m" → "-". Empty string in/out unchanged.

#### `parse_chord(s: str) → (int, str) | None`

Top-level chord parser. Strips whitespace, returns `None` for empty / `"UNK"` / `"NC"`. Otherwise applies `parse_root` then `normalize_quality`.

- *Input*: `"Am7"`.
- *Output*: `(9, "-7")`.

#### `parse_key(s: str) → int | None`

Extracts the tonic from a key string like `"Bb-maj"` or `"A-min"`. Splits at `-`, parses the leftmost letter.

- *Input*: `"G-maj"`.
- *Output*: `7`.

#### `transpose_chord(chord_string: str, semitones: int) → str`

Shifts a chord string by `semitones`. Returns the original string unchanged if unparseable.

- *Input*: `("Am7", 3)`.
- *Output*: `"Cm7"`.

#### `build_chord_tables(chord_vocab: list[str]) → (dict, dict, int, dict)`

Builds the lookup tables the model needs at training time.

- *Input*: `chord_vocab` — list of chord strings from `chord_vocab.json`.
- *Output*: tuple of `(chord_to_root, chord_to_qual, n_qualities, qual_to_id)`:
  - `chord_to_root[chord]` → root pitch class id (0..12).
  - `chord_to_qual[chord]` → quality id into the quality embedding.
  - `n_qualities` → total quality vocab size (including `__UNK_Q__`).
  - `qual_to_id[quality_string]` → quality id.

#### `chord_str_to_ids(chord_str: str, qual_to_id: dict) → (int, int)`

Resolves any chord string (including ones not in the training vocab) to `(root_id, quality_id)` with UNK fallback for both.

### 5.2 `model.py`

The PyTorch model.

**Module constants**

| Symbol           | Value | Meaning                                     |
|------------------|-------|---------------------------------------------|
| `TOK_REST`       | 0     | rest token                                  |
| `TOK_HOLD`       | 1     | sustain previous pitch                      |
| `TOK_NOTE_BASE`  | 2     | first NOTE_x token                          |
| `PITCH_LOW`      | 40    | lowest MIDI pitch the vocab covers          |
| `PITCH_HIGH`     | 96    | highest MIDI pitch the vocab covers         |
| `N_PITCHES`      | 57    | `PITCH_HIGH − PITCH_LOW + 1`                |
| `VOCAB_SIZE`     | 59    | `2 + N_PITCHES`                             |
| `TICKS_PER_BEAT` | 4     | 16th-note resolution in 4/4                 |
| `TICKS_PER_BAR`  | 16    | bar length in ticks                         |

**Class: `JazzLSTM(nn.Module)`**

Next-token predictor. At each tick the input is `(prev_token, chord_root, chord_quality, bar_position)`; the output is logits over the 59-token vocab.

#### `JazzLSTM.__init__(n_qualities, hidden=256, n_layers=2, dropout=0.1, d_token=64, d_root=8, d_qual=16, d_bar=8)`

Wires four embedding tables, an LSTM stack, and a linear head.

- `n_qualities`: size of the quality embedding (from `build_chord_tables`).
- `hidden`: LSTM hidden size.
- `n_layers`: stacked LSTM layers.
- `dropout`: inter-layer dropout (only effective when `n_layers > 1`).
- `d_*`: embedding dimensions for token / chord root / chord quality / bar position.

#### `JazzLSTM.forward(prev_token, chord_root, chord_qual, bar_pos, hidden=None) → (logits, hidden)`

- Inputs are `(B, T)` long tensors.
- Output `logits` is `(B, T, VOCAB_SIZE)`; `hidden` is the LSTM `(h, c)` tuple for chaining inference calls.

**Function: `load_model(checkpoint_path, device="cpu", quantize=True, jit=False) → (model, ckpt_dict)`**

Loads a checkpoint produced by `train.py`. Optionally applies dynamic int8 quantization on `nn.LSTM` and `nn.Linear` (the big Pi speed win, ~2-3×; no measurable accuracy loss) and TorchScript-compiles the result.

- *Returns*: `(model, ckpt)` with the model in eval mode. `ckpt` is the raw dict so callers can recover hyperparams.

### 5.3 `data_prep.py`

One-shot pipeline that converts the Weimar Jazz Database into the per-tick parquet `train.py` consumes.

**Module constants**

| Symbol             | Default                                       |
|--------------------|-----------------------------------------------|
| `CHORD_MIN_COUNT`  | `10` — chord vocab cutoff                     |
| `RAW_DIR`          | `"ML/data/raw"`                               |
| `OUT_DIR`          | `"ML/data/processed"`                         |
| `DB_PATH`          | `"ML/data/raw/wjazzd.db"`                     |
| `TOKEN_NAMES`      | `["REST","HOLD","NOTE_40",…,"NOTE_96"]`       |

**Functions**

#### `load_db() → (melody_df, beats_df, solo_info_df)`

Reads the three relevant tables from `wjazzd.db` as pandas DataFrames.

#### `filter_valid_solos(solo_info: DataFrame) → set[int]`

Keeps only 4/4 solos with a parseable `key` field. Returns set of `melid`s.

#### `prep_beats(beats: DataFrame) → DataFrame`

Forward-fills missing chord cells inside each solo, computes `beat_idx` (cumulative beat number) and `beat_dur` (seconds per beat, with the median-per-solo fallback on the trailing beat).

#### `align_notes(melody: DataFrame, beats: DataFrame) → DataFrame`

Joins melody onto beats. Computes:
- `frac`: fractional position within the beat (`0 ≤ frac < 1`).
- `beat_pos`: absolute float position in beats.
- `dur_beats`: note duration in beats.
- `tick`: integer 16th-note grid position (`round(beat_pos × 4)`).
- `dur_ticks`: integer duration in ticks (`round(dur_beats × 4)`, floor at 1).

#### `make_solo_grid(notes: DataFrame, beats_by_idx: dict) → DataFrame | None`

Per-solo encoder. Drops solos shorter than 10 notes. Sets the tick origin at the first note, pads up to a whole bar. Writes one row per tick:
- `tick`: 0..n-1.
- `token`: `REST` by default; on a note onset writes `NOTE_<pitch>`; subsequent ticks of the same note are `HOLD` (stops at the next note or at end).
- `chord`: chord at the beat covering this tick (forward-filled).
- `bar_pos`: `tick % 16`.

Returns the DataFrame, or `None` if too short.

#### `main()`

End-to-end driver. Filters, transposes every solo so the tonic = C (both pitches and chord roots shift by the same amount), drops notes outside `[PITCH_LOW, PITCH_HIGH]`, builds per-solo grids, computes the chord vocab (chords seen ≥10 times survive, the rest become `"UNK"`). Writes the four output files (`grid_solos.parquet`, `chord_vocab.json`, `token_vocab.json`, `metadata.json`) and prints a token distribution summary.

### 5.4 `train.py`

Training loop. CLI-driven.

**Module constants**

| Symbol             | Default                                       |
|--------------------|-----------------------------------------------|
| `DATA_DIR`         | `"ML/data/processed"`                         |
| `MODELS_DIR`       | `"ML/models"`                                 |
| `DEFAULT_CKPT_NAME`| `"jazz_lstm_v2.pt"`                           |

**Functions**

#### `load_solos(seq_len: int) → (list[dict], dict, int)`

Reads the parquet, attaches chord (root, quality) ids using `build_chord_tables`, groups by `melid`. Drops solos shorter than `seq_len + 1`. Each solo becomes a dict with numpy arrays `token`, `root`, `qual`, `bar`.

- *Returns*: `(solos, meta, n_qualities)`.

#### `sample_batch(solos, batch_size, seq_len, rng) → list[Tensor]`

Picks `batch_size` random fixed-length chunks across the corpus. For each chunk:

- `pt[i]` = tokens at positions `start .. start+T-1` (previous tokens fed into the LSTM).
- `cr[i]`, `cq[i]`, `bp[i]` = chord-root / chord-quality / bar-position at positions `start+1 .. start+T` (the *predicted* tick's features).
- `tg[i]` = tokens at `start+1 .. start+T` (the next-token targets).

Returns five tensors in that order, on CPU.

#### `main()`

CLI training driver. Args: `--epochs`, `--steps_per_epoch`, `--batch_size`, `--seq_len`, `--lr`, `--weight_decay`, `--grad_clip`, `--hidden`, `--n_layers`, `--dropout`, `--seed`, `--device`, `--out`. Builds `JazzLSTM`, optimises with Adam + cosine LR, clips gradients, writes the checkpoint to `ML/models/<out>` at the end of every epoch (overwriting). Saves alongside the weights: `vocab_size`, `n_qualities`, `hidden`, `n_layers`, `dropout`, `epoch`.

### 5.5 `engine.py`

The realtime runtime. Three top-level pieces matter: the mic aggregator, the sampler factory, and the `JazzImprov` engine itself.

**Module constants**

| Symbol                   | Value                                           |
|--------------------------|-------------------------------------------------|
| `TOK_REST`, `TOK_HOLD`   | re-imported from `model`                        |
| `TOK_NOTE_BASE`          | `2`                                             |
| `PITCH_LOW`, `PITCH_HIGH`| `40`, `96`                                      |
| `DEFAULT_CURRENT_PITCH`  | `60` — what `_current_pitch` is at reset        |
| `_HERE`                  | absolute path of `engine.py`                    |
| `DEFAULT_CHECKPOINT`     | `<_HERE>/models/jazz_lstm.pt` — **the knob to switch which model loads at inference** |
| `_VOCAB_PATH`            | `<_HERE>/data/processed/chord_vocab.json`       |

**Top-level helpers**

#### `pitch_to_note_token(midi_pitch: int) → int`

Inverse of `token_to_pitch`. Clamps to `[PITCH_LOW, PITCH_HIGH]` then maps to a `NOTE_x` token id.

#### `token_to_pitch(tok: int) → int | None`

Returns the MIDI pitch for a NOTE token; `None` for REST/HOLD.

#### `_note_name_to_midi(name: str) → int | None`

Parses strings like `"C#4"` or `"Bb3"` into MIDI numbers. Used by `MicTickAggregator.observe` when fed a note-name string instead of a numeric MIDI.

**Class: `MicTickAggregator`**

Collects per-callback pitch detections over a 16th-note window and emits one `REST`/`HOLD`/`NOTE_x` token at the tick boundary.

Module constants:

| Symbol             | Default               |
|--------------------|-----------------------|
| `DEFAULT_MIN_MIDI` | `48` (C3)             |
| `DEFAULT_MAX_MIDI` | `83` (B5)             |

#### `__init__(min_midi=None, max_midi=None)`

Sets the gate range and initialises the internal observation buffer and `_prev_pitch` tracker.

#### `_gate(midi) → int | None`

Returns `midi` if it's an integer inside `[min_midi, max_midi]`, otherwise `None`. Used to reject octave-error / nonsense detections.

#### `observe(midi_or_none) → None`

Called by the audio callback once per audio block. Accepts `None` (silence), `int`/`float` MIDI, or a note-name string. Appends the gated value (or `None`) to the internal list.

#### `consume_token() → int`

End-of-window resolver. Empties the internal list. Returns:
- `TOK_REST` if fewer than half the observations had a valid pitch (silence).
- `pitch_to_note_token(dominant)` if the dominant pitch differs from the previous tick's (new attack).
- `TOK_HOLD` if the dominant pitch matches the previous tick's (sustain).

Updates `_prev_pitch` to the dominant pitch (or `None` for silence).

**Function: `make_sampler(...) → callable`**

#### `make_sampler(temperature=1.0, hold_penalty=0.0, rest_penalty=0.0, max_consec_holds=None, rng=None) → (logits, consec_holds=0) → int`

Builds a sampling closure with fixed hyperparams. The returned `sample(logits, consec_holds)` function:

1. Subtracts `hold_penalty` from the HOLD logit.
2. Subtracts `rest_penalty` from the REST logit.
3. Sets the HOLD logit to `−∞` if `consec_holds >= max_consec_holds` (hard cap that prevents arbitrarily long sustains).
4. Divides by temperature, softmaxes, samples a single token id.

**Class: `JazzImprov`**

The realtime engine. State is three groups:

- **Persistent**: `h_persistent`, `prev_actual`, `current_tick`, `_current_pitch`, `_prev_pitch` — the actual player history.
- **Rollout**: `_rollout_buf`, `_h_rollout`, `_prev_rollout_tok`, `_rollout_next_tick` — the model's prediction for the upcoming ~30 ticks.
- **Lock counter**: `_locked_ticks` — N ticks during which player input is forced to HOLD after a deviation.

#### `__init__(model, *, temperature=1.0, rollout_ticks=30, deviation_lock_ticks=1, hold_penalty=1.0, rest_penalty=1.5, max_consec_holds=None, deviation_initial_fresh=2, rollout_extend_per_tick=4, seed=0, device="cpu")`

Stores hyperparams, builds the sampler. The sampler hyperparams are also recorded in `_sample_params` so a future `clone()` could rebuild a divergent sampler (the clone method is not currently present; see §10). `chord_root`/`chord_qual`/`bar_chords`/`total_ticks` are initialised to `None` until `set_progression` is called.

- `temperature`: float, sampling temperature. 0.4 ballad, 0.9 bebop.
- `rollout_ticks`: target rollout length. 30 = ~5.6s lookahead.
- `deviation_lock_ticks`: total length (in ticks) of the post-deviation lock period. Internally `n_lock = deviation_lock_ticks − 1` HOLDs are forced.
- `hold_penalty`, `rest_penalty`: see `make_sampler`.
- `max_consec_holds`: hard cap on consecutive HOLDs.
- `deviation_initial_fresh`: tokens sampled fresh on deviation (rest filled by catchup).
- `rollout_extend_per_tick`: max tokens the catchup extends per commit.
- `seed`: RNG seed.

#### `current_pitch` (property) → `int`

The pitch the engine currently believes the player is on.

#### `is_done` (property) → `bool`

`current_tick >= total_ticks`.

#### `set_progression(chord_progression: list[(str, int)], total_bars: int, qual_to_id: dict) → None`

Lays out the per-tick `chord_root` and `chord_qual` arrays. `chord_progression` is `[(chord_str, n_bars), …]`; it's cycled to fill `total_bars`. Each chord covers `TICKS_PER_BAR = 16` consecutive ticks. Stores `bar_chords` (list of length `total_bars`) and `total_ticks`.

#### `reset() → None`

Resets all runtime state to a fresh init: `current_tick = 0`, `prev_actual = TOK_REST`, no hidden state, empty rollout, `_current_pitch = DEFAULT_CURRENT_PITCH`, `last_action = "init"`. Then calls `_refresh_rollout()` so a fresh 30-token rollout exists before the first `commit()`.

#### `_forward_one(prev_token: int, t: int, hidden) → (logits, new_hidden)`

One LSTM step that predicts the token at tick `t`. Pulls `chord_root[t]`, `chord_qual[t]`, computes `bar_pos = t % TICKS_PER_BAR`, runs `model(...)`. `prev_token` is the previously-committed/sampled token id.

#### `_clone_hidden(h) → hidden_or_None` *(staticmethod)*

Deep-copies an LSTM hidden tuple `(h, c)`. Used so the rollout can fork off the persistent state without aliasing.

#### `_count_trailing_holds(buf: list[int]) → int` *(staticmethod)*

Counts how many `TOK_HOLD` tokens are at the *end* of a buffer. Used to seed `consec_holds` for the sampler so the hard cap respects what's already in the rollout.

#### `_token_to_pitch(tok: int) → int`

Resolves a token to a MIDI pitch: NOTE_x → its pitch, HOLD/REST → `_current_pitch`. Used by the match check.

#### `_update_current_pitch(tok: int) → None`

If `tok` is a NOTE token and its pitch differs from `_current_pitch`, sets `_prev_pitch = _current_pitch` then `_current_pitch = new pitch`. Otherwise no-op (HOLD/REST don't update either field).

#### `_next_expected_pitch() → int | None`

Walks `_rollout_buf` until it finds a pitch different from `_current_pitch`. Returns that pitch, or `None` if the whole rollout sits on the current pitch. Used by the tolerance check.

#### `_sample_n(h, prev, n, start_tick, consec_holds=0) → (tokens, h, prev, consec_holds)`

Samples `n` tokens starting from hidden `h`, previous token `prev`, beginning at engine tick `start_tick`. Returns the sampled tokens, the final hidden state, the final previous token, and the trailing consec-hold count. Used by every rollout-building path (`_refresh_rollout`, `_extend_*`, `_build_deviation_rollout`).

#### `_refresh_rollout() → None`

Builds the rollout from scratch using a clone of the persistent state. Called only at `reset()`.

#### `_extend_rollout_by_one() → None`

Samples one new token at `_rollout_next_tick`, appends to the buffer. No-op past `total_ticks`. Used by every normal commit path.

#### `_extend_rollout_by_n(n: int) → None`

Batched version: samples `n` tokens with a single `_sample_n` call. Faster than n separate calls.

#### `_catchup_extend() → None`

If the rollout is shorter than `rollout_ticks`, calls `_extend_rollout_by_n(min(deficit, rollout_extend_per_tick))`. Runs at the end of the match / silence / locked commit paths to refill a rollout that was shortened by a previous deviation.

#### `_build_deviation_rollout(n_lock: int, initial_fresh: int | None = None) → None`

Rebuilds the rollout starting with `n_lock` forced HOLDs (visual sustain of the deviation note), followed by `initial_fresh` fresh sampled tokens. `initial_fresh` defaults to `self.deviation_initial_fresh`. Walks the LSTM through the lock without sampling so the post-lock state reflects "the player has been holding the deviation pitch for `n_lock` ticks." `consec_holds` is seeded at `n_lock` so the post-lock samples respect the max-consec cap.

#### `_advance_persistent(played_token: int) → None`

Runs one LSTM forward with the current `prev_actual`, stores the new hidden as `h_persistent`, sets `prev_actual = played_token`, increments `current_tick`. This is the *only* place `h_persistent` and `current_tick` change.

#### `_start_deviation(played_token: int) → None`

Common path for both "fresh deviation" and "deviation during lock". Advances persistent state with `played_token`, updates current pitch, rebuilds the deviation rollout with `n_lock = max(0, deviation_lock_ticks − 1)`, sets `_locked_ticks = n_lock`, sets `last_action = "deviation"`.

#### `commit(played_token: int) → list[int]`

The public per-tick driver. Resolves three cases:

1. **Done**: `current_tick >= total_ticks` → sets `last_action = "done"`, returns the current rollout snapshot.
2. **Locked**: `_locked_ticks > 0`:
   - If the played token is a NOTE attack at a new pitch → break the lock with `_start_deviation`.
   - Otherwise continue the lock: advance persistent with `TOK_HOLD`, pop the rollout front, extend by one, catchup-extend, decrement `_locked_ticks`. `last_action = "locked"`.
3. **Normal**:
   - If REST and rollout is non-empty → silence-as-match: replace `played` with `_rollout_buf[0]`, set `matched = True`, `last_action = "silence"`.
   - Otherwise compute `expected_pitch` from `rollout[0]`, `played_pitch` from `played`, and the *tolerance buffer*: `matched = played_pitch ∈ {expected_pitch, _prev_pitch, _next_expected_pitch()}`.
   - Advance persistent with `played` and update current pitch.
   - If matched: pop the rollout front, extend by one, catchup-extend, `last_action = "match"` (unless it was already "silence").
   - If not matched: `n_lock = max(0, deviation_lock_ticks − 1)`, call `_build_deviation_rollout(n_lock)`, set `_locked_ticks = n_lock`, `last_action = "deviation"`.

Returns the rollout snapshot as a fresh list.

#### `rollout() → list[int]`

Read-only copy of `_rollout_buf`.

**Function: `make_engine(chord_progression, *, total_bars=64, temperature=1.0, rollout_ticks=30, deviation_lock_ticks=1, hold_penalty=1.0, rest_penalty=1.5, max_consec_holds=None, deviation_initial_fresh=2, rollout_extend_per_tick=4, checkpoint_path=None, vocab_path=None, quantize=True, jit=False, seed=None, device="cpu") → JazzImprov`**

Convenience factory. Loads the checkpoint from `DEFAULT_CHECKPOINT` (or override), reads the chord vocab JSON, builds `qual_to_id`, instantiates `JazzImprov`, calls `set_progression` and `reset`. Quantize=True is the default and is the big Pi performance lever.

### 5.6 `generate.py`

Offline MIDI rendering CLI + re-exports for older import paths.

**Re-exports**: `JazzImprov`, `MicTickAggregator`, `pitch_to_note_token`, `token_to_pitch`, `make_engine as make_realtime_engine`, and the token/pitch constants. So any code that does `from generate import …` keeps working.

**Module constant**

`_CHORD_INTERVALS` — dict mapping normalized chord qualities to interval lists used for the comping voicing. Covers the common qualities ("", "j7", "7", "-", "-7", "o7", "sus", …).

**Functions**

#### `_tokens_to_events(tokens: list[int]) → list[(onset_tick, midi, dur_ticks)]`

Walks a token stream and emits `(onset, midi, duration)` per NOTE event. Skips leading REST/HOLD; each NOTE_x starts an event whose duration is `1 + count of trailing HOLDs`.

#### `_chord_voicing(chord_str: str, bass_min=36, upper_min=48) → list[int]`

Returns a list of MIDI pitches: one bass note at or above `bass_min` plus all chord-quality intervals voiced at or above `upper_min`. Used to generate the comping MIDI track. Unparseable chord → `[]`.

#### `_vlq(n: int) → bytes`

MIDI variable-length-quantity encoder. Used inside `write_midi`.

#### `_make_track(payload: bytes) → bytes`

Wraps a MIDI track payload in the `MTrk` chunk with a length prefix.

#### `write_midi(tokens, bar_chords, out_path, tempo_bpm=80, ticks_per_quarter=480) → None`

Renders a token stream + chord progression to a standard MIDI file with three tracks (tempo, melody on channel 0, chord comping on channel 1). `tokens` is the engine's full committed token sequence; `bar_chords` is the chord-per-bar list from `engine.bar_chords`.

#### `main()`

CLI. Args: `--chords` (comma-separated progression in C), `--bars`, `--temperature`, `--hold_penalty`, `--rest_penalty`, `--max_consec_holds`, `--seed`, `--tempo_bpm`, `--out`. Builds a realtime engine and commits TOK_REST until done (the silence-as-match path lets the engine self-play). Writes the result to MIDI.

### 5.7 `perf.py`

Lightweight timing buckets. Used by `Piano_display.py` to measure `engine.commit()` cost (separately for match / silence / deviation / locked) on the Pi.

**Class: `PerfLogger`**

#### `__init__(name="perf")`

Initialises an empty `buckets` dict. Each bucket's value is a list of duration samples in milliseconds.

#### `start() → float`

Returns a starting timestamp (`time.perf_counter()`).

#### `stop(t0: float, bucket: str) → float`

Records `(now − t0) * 1000` into `bucket`. Returns the duration in milliseconds.

#### `record(bucket: str, dt_ms: float) → None`

Appends a raw duration into a bucket, creating the bucket if needed.

#### `timed(bucket: str) → _Timed`

Context-manager wrapper:

```python
with perf.timed("commit"):
    engine.commit(...)
```

#### `clear() → None`

Drops all recorded samples.

#### `_stats(samples: list[float]) → dict` *(staticmethod)*

Returns `{n, mean, p50, p95, p99, max}` in milliseconds. `None` for empty input.

#### `summary(recent_n: int | None = None) → dict`

Per-bucket stats dict. Optionally restricts to the last `recent_n` samples per bucket.

#### `print_summary(recent_n=None, header=True) → None`

Prints a formatted table to stdout. Buckets are sorted with `deviation` first so the spike stands out.

#### `save_csv(path: str) → None`

Dumps every raw sample to CSV: `bucket, sample_idx, duration_ms`.

#### `save_summary_csv(path: str, recent_n=None) → None`

Dumps the aggregated summary (one row per bucket) to CSV.

**Class: `_Timed`** *(internal)*

Context manager returned by `PerfLogger.timed()`. `__enter__` records the start time; `__exit__` calls `record` on the parent logger.

---

## 6. SP/ — signal processing

### 6.1 `signalprocessing.py`

Realtime pitch detection. One audio block at a time, FFT + Harmonic Product Spectrum + spectral-flux onset detector. Exposes `detector.last_midi` (int or `None`), `detector.melody` (rolling 10-note name array), `detector.keyboard` (88-key on/off mask).

**Module constants**

| Symbol               | Value                                              |
|----------------------|----------------------------------------------------|
| `SAMPLE_RATE`        | `44100`                                            |
| `FFT_SIZE`           | `4096` (~10.8 Hz/bin)                              |
| `HOP_SIZE`           | `512` (~11.6 ms/callback)                          |
| `A4_FREQ`            | `440.0`                                            |
| `SILENCE_THRESH`     | `0.02` RMS threshold                               |
| `LO_HZ`, `HI_HZ`     | `27.5, 4200.0` — search range A0 .. C8             |
| `HPS_HARMONICS`      | `5`                                                |
| `CONFIRM_FRAMES`     | `2` — frames a pitch must stabilise before emit    |
| `FLUX_HISTORY`       | `22` — frames of spectral-flux history for adaptive onset |
| `FLUX_MULT`          | `1.2` — onset threshold = `mult × median(history)` |
| `OCTAVE_DOWN_THRESH` | `0.5` — half-frequency ratio that triggers octave-down correction |
| `OCTAVE_DOWN_MAX`    | `3` — max octaves to descend per detection         |

Plus a 4-pole Butterworth bandpass (`27.5 .. 4200` Hz) as a stateful SOS filter.

**Helpers**

#### `freq_to_midi(freq: float) → int`

`round(12 × log2(freq / 440) + 69)`.

#### `midi_to_name(m: int) → str`

`"C4"`, `"F#3"`, etc.

#### `parabolic_interp(spectrum: np.ndarray, idx: int) → float`

Sub-bin peak refinement. Returns the fractional bin index for the true peak around `idx`.

#### `hps(mag_full: np.ndarray, harmonics: int = HPS_HARMONICS) → np.ndarray`

Harmonic Product Spectrum: multiplies the magnitude spectrum by `harmonics − 1` downsampled copies, so peaks at fundamentals reinforce each other. Returns an array of length `len(mag_full) // harmonics`.

**Class: `PitchDetector`**

#### `__init__()`

Allocates the FFT ring buffer, IIR filter state, normalised-magnitude history (for spectral flux), flux history, candidate-stabilisation state, melody/keyboard outputs.

#### `_filter(block: np.ndarray) → np.ndarray`

Applies the stateful bandpass filter so block boundaries don't introduce clicks.

#### `_update_onset(mag_norm: np.ndarray) → bool`

Computes spectral flux as the sum of positive bin-to-bin differences between this block's normalised magnitude and the previous one's. Updates the rolling history. Returns `True` if `flux > FLUX_MULT × median(non-zero history)`.

#### `_detect_pitch(mag_full: np.ndarray) → float | None`

The HPS pipeline:

1. Run HPS on the full magnitude.
2. Constrain to the search range.
3. Parabolic interpolation → fractional bin → Hz via linear interp on `FREQS`.
4. Reject if outside MIDI `[21, 108]`.
5. *Iterated octave-down correction*: if `mag_full[peak_bin / 2] >= OCTAVE_DOWN_THRESH × mag_full[peak_bin]`, halve the frequency and repeat (up to `OCTAVE_DOWN_MAX` times). Catches HPS lock-ons to harmonics on bright sounds.

Returns the corrected fundamental in Hz, or `None`.

#### `callback(indata, frames, time_info, status) → None`

The sounddevice audio callback. Per audio block:

1. RMS silence gate (resets `last_midi` to `None` and returns if below threshold).
2. Bandpass filter, scroll into the FFT ring buffer.
3. Full magnitude FFT, normalised magnitude.
4. Onset detector update.
5. Pitch detection.
6. Candidate-stabilisation: only commit a MIDI value after `CONFIRM_FRAMES` consecutive blocks at that pitch.
7. **Emit only on a confirmed onset**: spectral flux is the sole trigger. `self.last_midi` is updated to the new pitch, the rolling melody is shifted, the keyboard mask is set, and a trace line is printed.

So `last_midi` stays at the last attacked pitch through a sustained note, then clears to `None` on silence, then updates on the next attack.

**Function: `main()`**

Standalone demo. Opens a sounddevice `InputStream` and prints detections until Ctrl-C.

---

## 7. HW/bap/ — hardware-side inputs

### 7.1 `Buttons_v2.py`

Chord-progression entry UI. Pygame window with the current chord and progression rendered as text; keyboard input for chord selection.

**Constants** (defined inside `main()`)

- `chord_dict` — `{0: "A", 1: "A#", …, 11: "G#"}` (root letters in order, w-key direction).
- `mod_dict` — `{0: "", 1: "m", 2: "sus2", …, 14: "min9"}` (chord-quality suffixes).

**Function: `main() → list[str]`**

Opens a 1920×1080 fullscreen pygame window. Loops:

- `w` / `s` — root up/down (mod 12).
- `↑` / `↓` — quality up/down (mod 15).
- `c` — append current chord (e.g. `"Am7"`) to the progression, reset chord to `[0, 0]`.
- `p` — preset progression `["Am7","Bm7","E9","E9"]`.
- `d` — confirm: quits pygame and returns the progression list.
- `q` / quit — sets `done` (also quits and returns the list).

Returns the chord progression as a Python list of strings.

### 7.2 `bapv1.py` *(deprecated)*

The old pitch detector. Kept only because the legacy `inputty`/`check_note` scoring path referenced it. The current code path uses `signalprocessing.py` exclusively; `bapv1` is no longer imported by `Piano_display.py`.

**Module constants**: `SAMPLE_RATE = 44100`, `FFT_SIZE = 2048`, `SILENCE_THRESH = 0.02`, `A4_FREQ = 440.0`, plus a precomputed Hanning window and 100-tap FIR bandpass.

**Class: `PureArrayDetector`**

Older single-FFT detector. Maintains a ring buffer, computes RMS for silence gating, runs the FFT, finds the spectral peak within `[20, 4200]` Hz, converts to MIDI via `freq_to_midi`. Exposes `last_midi`, `last_note`, `melody` (a Python list, appended with each new note), `keyboard` (88-key mask). No HPS, no spectral-flux onset, no octave correction.

---

## 8. `Piano_display.py` — main application

The pygame frontend. Pulls a chord progression from `Buttons_v2`, sets up the realtime engine, opens the mic stream, then runs a 60-FPS pygame loop.

### Module-level constants

| Symbol                          | Default                                                         |
|---------------------------------|-----------------------------------------------------------------|
| `SAMPLE_RATE`, `HOP_SIZE`       | from `signalprocessing` (44100, 512)                            |
| `WIDTH`, `HEIGHT`               | `36`, `20` — visible columns / rows                             |
| `fps`                           | `60`                                                            |
| `bpm`                           | `60` (user-tuneable)                                            |
| `Testing_variable_for_testing`  | `fps * fps // bpm // 4` — frames per 16th note                  |
| `speed`                         | `1 / Testing_variable_for_testing` — y-units per frame          |
| `detector`                      | `signalprocessing.PitchDetector()` instance                     |
| `buffer_sheet`                  | `[None]*15 + [17]` — 16-row pre-roll buffer                     |
| `col_start`                     | `48` — MIDI for visual column 0                                 |
| `Note_list`                     | 36-element list of note name strings, one per column            |
| `engine`                        | the `JazzImprov` instance for the current run                   |
| `mic_aggregator`                | the `MicTickAggregator` instance                                |
| `PITCH_LOW_DISPLAY`, `PITCH_HIGH_DISPLAY`, `DEFAULT_DISPLAY_PITCH` | `48`, `83`, `60` — display-side pitch clipping for the render chain |
| `_PLAY_LINE_OFFSET`             | `31` — engine_tick at play line = `tally − 31`                  |
| `total_sheet`                   | the buffered + initial-rollout per-tick column list             |
| `DEBUG_REALTIME`                | `True` — turns on per-tick log lines                            |
| `PERF_PRINT_EVERY_N_TICKS`      | `100`                                                           |
| `PERF_RECENT_WINDOW`            | `100`                                                           |
| `PERF_CSV_ON_QUIT`              | `"commit_timings.csv"`                                          |
| `perf_commit`, `perf_tick_block`| `PerfLogger` instances                                          |
| `hor_pos`, `key_width`          | 37-element lists used to position notes against the visual keyboard |

### Functions

#### `midi_to_col(midi: int) → int | None`

`midi − col_start`; returns `None` for pitches outside `[col_start, col_start + 35]`.

#### `render_tokens(tokens: list[int], start_prev_midi: int) → (list[int|None], int)`

Walks a token stream and produces one column index per tick. REST/HOLD reuse the previous pitch (we're monophonic and can't draw "rest" without a glyph). NOTE_x updates `prev` to the new pitch, clamped to `[PITCH_LOW_DISPLAY, PITCH_HIGH_DISPLAY]`. Returns `(cols, final_prev_midi)` so the caller can chain runs.

#### `_detector_callback_with_aggregation(indata, frames, time_info, status) → None`

Wraps the original `detector.callback`. After running the original (which updates `detector.last_midi`), calls `mic_aggregator.observe(detector.last_midi)`. Installed before `sounddevice.InputStream` is started so the stream uses the wrapper.

#### `get_played_token() → int`

`mic_aggregator.consume_token()`. Called once per tick boundary.

#### `chord_at_play_line(tally: int) → str`

Chord at the figure currently on the play line. `engine_tick = max(0, tally − _PLAY_LINE_OFFSET)`, then `engine.bar_chords[engine_tick // 16]`. Empty string before the first commit.

#### `chord_upcoming(tally: int) → str`

The next bar's chord — `engine.bar_chords[engine_tick // 16 + 1]`, clamped to the last bar.

### Classes

#### `Figure`

One per 16th-note tick. Carries the `source_idx` (the tally at spawn) so the engine can later find this figure by source_idx and replace its content. Internally just stores the column index and the pre-computed note name.

- `__init__(col, source_idx=None)`
- `col_idx` (property) — `int 0..35` or `None`.
- `note_name` (property) — `str` like `"C4"` or `""`.
- `update_image(col)` — replace the figure's column after a deviation re-plan.

#### `BeatBar`

A horizontal bar-divider line. `y` and an optional `label` (the upcoming chord name when it changes from the previous bar).

- `__init__(label="")`

#### `Music`

Holds the on-screen state: a deque of `Figure` objects, a list of `BeatBar` objects.

- `__init__(height, width)` — `x = 40`, `y = 40`, `zoom = 50`, empty `figure` deque, empty `beat_bars`.
- `new_figure(col, source_idx=None)` — appends a `Figure` to `self.figure`.
- `new_beatbar(label="")` — appends a `BeatBar` to `self.beat_bars`, drops the oldest if more than 3.
- `go_down()` — increments `y` of every figure and bar by `speed` (called every frame).

### The main loop

Three sources of work, gated by `counter` (incremented every frame):

1. **Per-tick block** (every `Testing_variable_for_testing` frames):
   - Sentinel `_t_block = None` (the perf-logger uses this to know whether it should record a tick-block sample).
   - If `tally >= 31` and not done:
     - `perf_commit.start()`.
     - `played_token = get_played_token()`.
     - `engine.commit(played_token)` (measured into the perf logger, bucketed by `engine.last_action`).
     - Render the new rollout into per-tick column indices with `engine.current_pitch` as the prev-pitch seed.
     - **Update on-screen figures** for `k = source_idx − tally + 14` in `[0, min(14, len(rollout_cols)))`. This is what makes deviations visibly land near the play line.
     - **Replace/extend `total_sheet`** from `rollout_cols[14:]` starting at index `tally` — upcoming spawn content.
   - Spawn the next figure from `total_sheet[tally]`, increment `tally`.
   - Pop the oldest figure if it's scrolled past `y > 19` or is silent (`col_idx is None`).
   - Close the perf-tick-block timer if it was started.
   - Periodic `perf_commit.print_summary` every `PERF_PRINT_EVERY_N_TICKS`.

2. **Per-bar block** (every `Testing_variable_for_testing * 16` frames):
   - Compute `bar_idx = (tally − 16) // 16` (the bar that the new line will mark at the play line in 15 ticks).
   - Compare to the previous bar's chord; pass the chord name as `label` if it differs from the previous bar, else empty string.
   - `game.new_beatbar(chord_label)`.

3. **Every frame**:
   - `game.go_down()` — scrolls all figures and bar lines.
   - Pygame event handling (`K_r` resets `Music`; `K_ESCAPE`/`K_q` quits).
   - Blit background, draw figures (grouping consecutive same-column figures into single tall rectangles in green), draw bar lines (with chord labels in the small `font_bar` at the left margin if the bar has a label), draw the top-row text (BPM, Time, Cur note, Chord, Next chord), draw the blue keyboard marker for the currently-detected pitch.
   - `pygame.display.flip()`, `clock.tick(fps)`.

On quit: print the final perf summary, write `commit_timings.csv` and `commit_timings_summary.csv` if `PERF_CSV_ON_QUIT` is set, `pygame.quit()`.

### Display geometry

- The playing area starts at `(game.x, game.y) = (40, 40)`, spans `WIDTH * zoom = 1800` pixels wide and `HEIGHT * zoom = 1000` pixels tall in tick coordinates.
- Figures spawn at `y = 0` (top), scroll downward at `speed`, reach the play line at `y = 15`, and are popped after `y > 19`.
- Bar lines follow the same coordinate system.
- The red play line is drawn at pixel `game.y + 15 * zoom`.

### Timing alignment

With the 16-row buffer:

- `total_sheet[16]` is the first real engine prediction (engine tick 0).
- A figure with `source_idx = 16` reaches `y = 15` at wall-clock tally `31`.
- The engine's first commit fires at tally `31` (`tally >= 31` gate), and it advances `current_tick` from 0 to 1 — i.e. it commits engine tick 0.
- So the figure crossing the play line = the engine tick the engine is committing. The chord readout uses `_PLAY_LINE_OFFSET = 31` so it shows the chord at the same engine tick.

### Bar-line labels

A new bar line spawned at tally K reaches the play line at tally K+15. With the 16-row buffer, that lines up with engine tick K−16 crossing the play line, i.e. the start of bar `(K − 16) // 16`. We label the line with that bar's chord — but only if it differs from the previous bar — so the player gets a heads-up for actual chord changes.

---

## 9. Operational notes

### Switching which model loads

Edit one line in `ML/engine.py`:

```python
DEFAULT_CHECKPOINT = os.path.join(_HERE, "models", "jazz_lstm.pt")
```

Change the filename to whichever `.pt` you want to use at inference. `make_engine` (and therefore `make_realtime_engine`) read this constant unless explicitly overridden by `checkpoint_path=...`.

### Training a new model

```powershell
python ML\train.py `
  --out jazz_lstm_v3.pt `
  --hidden 384 --n_layers 3 `
  --dropout 0.2 `
  --batch_size 128 --seq_len 512 `
  --epochs 100 --steps_per_epoch 500 `
  --lr 1e-3 --weight_decay 1e-4
```

The checkpoint is saved to `ML/models/<--out>` at the end of every epoch (overwriting); interrupting mid-training keeps the last completed epoch. Use the `epoch` field inside the saved dict to check progress.

### Re-running data prep

```powershell
python ML\data_prep.py
```

Reads `ML/data/raw/wjazzd.db`, writes `ML/data/processed/*`. Only needed if you change the encoding, the chord vocab threshold, or the transposition logic.

### Tuning the realtime engine

In `Piano_display.py`, the `make_realtime_engine(...)` call exposes:

- `temperature` — 0.4 ballad ↔ 0.9 bebop.
- `rollout_ticks` — total rollout length (= visual lookahead). 30 ticks ≈ 5.6 s at 140 BPM.
- `deviation_lock_ticks` — total length of the post-deviation lock period (held quarter etc.).
- `hold_penalty`, `rest_penalty` — soft logit biases.
- `max_consec_holds` — hard upper bound on note length.
- `deviation_initial_fresh` — tokens sampled fresh on a deviation tick. Higher = more of the new plan appears instantly but a larger one-tick LSTM spike.
- `rollout_extend_per_tick` — how aggressively catchup refills the rollout in subsequent ticks.

The two amortisation knobs trade off "spike on deviation" vs "wave-of-change visual after deviation."

### Reading the perf log

The terminal periodically prints e.g.:

```
  16th-note budget at 60 BPM = 250.0 ms

── engine.commit timings (ms) (last 100) ──
  bucket              n    mean     p50     p95     p99     max
  deviation           7   42.18   41.20   48.55   48.55   48.55
  locked             18    2.31    2.21    2.85    3.04    3.04
  match              31    2.58    2.50    3.42    3.61    3.61
  silence            44    2.41    2.31    3.10    3.27    3.27
```

The number to watch is `deviation.max`. If it approaches the 16th-note budget (`60_000 / bpm / 4` ms), the engine is at risk of dropping ticks. Cheapest mitigation: lower `rollout_ticks` (covers fewer LSTM forwards per deviation) or raise `deviation_initial_fresh` and reduce `rollout_extend_per_tick` (already amortised).

---

## 10. Parked decisions / known artifacts

### Inference-time transposition

Training data is C-tonic. The runtime feeds the user's chord progression directly into `chord_str_to_ids` with no key shift, so the model "plays in C" regardless of the user's intended key. `Cmaj7` alone sounds idiomatic; `Am7 Bm7 E9 E9` sounds wrong because the model treats those as non-C-tonic chords sitting in a C-major context. Fix would be: detect/specify key, shift chord roots to C before `set_progression`, shift the detector input up by the same amount, shift the rendered output back. Not implemented yet.

### Deviation visual wave during catchup

`deviation_initial_fresh = 2` means the deviation tick only paints 5 figures (k = 0..4 from the play-line up). The remaining on-screen figures (k = 5..13) retain old-plan content until `_catchup_extend()` reaches them in the following few ticks. So a "wave of change" propagates outward from the play line over ~5 ticks. To pin every on-screen figure to the new plan immediately, set `deviation_initial_fresh = 11` (covers all 14 on-screen slots); the deviation tick cost rises from ~6 to ~15 LSTM forwards but the visual wave goes away.

### `bapv1.py` deprecated

No longer imported anywhere. Kept on disk for now in case there's a downstream tool still pointing at it.

### Double-timeline feature

Tried in an earlier iteration; not currently in the codebase. The remnants (`_prev_pitch` field) of the tolerance-buffer machinery the double-timeline code shared are still in place because they're useful for the single timeline too.

### Off-by-one in the rollout → figure update mapping

The on-screen update uses `k = source_idx − tally + 14` (not `+ 15`). The constant was carried over from an earlier prototype and works empirically. If you ever rewrite the timing, the principled formula is `engine_tick(figure) = source_idx − 16` and `engine_tick(rollout[k]) = current_tick + k`, with `current_tick = tally − 30` post-commit; solve for k.
