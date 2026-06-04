"""Realtime glue between the JazzImprov engine and the static-sheet
Piano_display.

`Piano_display` reads a `sheet = [(col, length), ...]` list and spawns one
figure per entry, waiting `length` ticks between spawns. This module owns
the engine, the mic aggregator, and the sheet — keeps the sheet up to date
each 16th-note tick by:

    1. consuming player input from the SP aggregator,
    2. committing it to the engine (advances persistent state, possibly
       triggers a deviation rebuild of the rollout),
    3. refreshing the *future* tail of the sheet from the new rollout while
       leaving already-spawned events alone.

So when the player matches the plan the sheet stays stable; when they
deviate, the upcoming spawns reflect the model's new plan.

Wiring from Piano_display.py is small:

    from realtime_sheet import RealtimeSheet
    rs = RealtimeSheet(chord_prog_2, total_bars=64)
    sheet = rs.sheet                 # share the list reference

    # Hook the SP detector callback so every audio block feeds the aggregator
    _orig = detector.callback
    def _hooked(indata, frames, time_info, status):
        _orig(indata, frames, time_info, status)
        rs.aggregator.observe(detector.last_midi)
    detector.callback = _hooked

    game = Music(HEIGHT, WIDTH, sheet)
    rs.attach_music(game)            # rs reads game.note for spawn progress

    # In the per-16th-note block of the main loop:
    if counter % (Testing_variable_for_testing) == 0:
        rs.tick()                    # ← only new line in the inner loop
        game.new_figure()
        ...
"""

from engine import (
    make_engine, MicTickAggregator,
    TOK_REST, TOK_HOLD, TOK_NOTE_BASE,
    PITCH_LOW,
)


# ----------------------------------------------------------------------
# Token sequence -> discrete sheet events
# ----------------------------------------------------------------------

def tokens_to_events(tokens, col_start=48):
    """Convert a per-tick token sequence into discrete display events.

    Each NOTE_x starts a new event with length = 1 + (count of following
    HOLDs). REST and any leading HOLDs (before the first NOTE_x) are
    skipped. Returns list of `[col_index, length]` lists (mutable, since
    Piano_display indexes them like list-of-lists).
    """
    events = []
    i, n = 0, len(tokens)
    while i < n:
        tok = int(tokens[i])
        if tok >= TOK_NOTE_BASE:
            midi = PITCH_LOW + (tok - TOK_NOTE_BASE)
            col = max(0, midi - col_start)
            length = 1
            j = i + 1
            while j < n and int(tokens[j]) == TOK_HOLD:
                length += 1
                j += 1
            events.append([col, length])
            i = j
        else:
            i += 1
    return events


# ----------------------------------------------------------------------
# The realtime sheet controller
# ----------------------------------------------------------------------

class RealtimeSheet:
    """Drives a JazzImprov engine per 16th-note tick and exposes a
    Piano_display-compatible `sheet` of [col, length] events.

    State:
      * `self.engine`     — JazzImprov instance
      * `self.aggregator` — MicTickAggregator (hook the SP callback into it)
      * `self.sheet`      — the list Music reads; mutated in place
      * `self._music`     — attached Music (so we can read its spawn counter)
    """

    # Reasonable realtime defaults — overridable via __init__ kwargs.
    DEFAULTS = dict(
        temperature=0.45,
        rollout_ticks=30,
        deviation_lock_ticks=4,        # quarter-note lock on deviation
        hold_penalty=0.7,
        rest_penalty=1.5,
        max_consec_holds=4,            # cap notes at quarter
        quantize=True,                 # int8 LSTM/Linear for Pi performance
    )

    def __init__(self, chord_progression, *, total_bars=64,
                 col_start=48, debug=False, **engine_kwargs):
        """Args:
            chord_progression : list of (chord_str, n_beats) tuples.
                                Each chord lasts `n_beats` quarter notes.
                                Internally converted to bars for the engine.
            total_bars        : how many bars total in the song.
            col_start         : MIDI value at sheet column 0 (default 48 = C3
                                — matches Piano_display.col_start).
            debug             : if True, print per-tick action + sheet head.
            **engine_kwargs   : forwarded to `make_engine`. Override the
                                DEFAULTS above (e.g. `temperature=0.6`).
        """
        self.col_start = int(col_start)
        self.debug = bool(debug)

        # Convert beats -> bars for the engine (4 beats/bar in 4/4)
        bar_prog = []
        for chord, n_beats in chord_progression:
            n_beats = max(1, int(n_beats))
            n_bars = max(1, (n_beats + 3) // 4)
            bar_prog.append((chord, n_bars))

        kwargs = dict(self.DEFAULTS)
        kwargs.update(engine_kwargs)
        self.engine = make_engine(bar_prog, total_bars=total_bars, **kwargs)
        self.aggregator = MicTickAggregator()

        # Initial sheet = whatever the engine predicted before anyone played
        self.sheet = tokens_to_events(self.engine.rollout(), self.col_start)

        self._music = None
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach_music(self, music):
        """Connect the Music instance so we know which sheet events have
        been spawned (and therefore can't be rewritten on deviation)."""
        self._music = music

    @property
    def spawned_count(self):
        """Number of sheet events Piano_display has already spawned."""
        return self._music.note if self._music is not None else 0

    # ------------------------------------------------------------------
    # The per-tick loop
    # ------------------------------------------------------------------
    def tick(self):
        """Call exactly once per 16th-note tick. Reads player input from
        the aggregator, commits to the engine, refreshes the future tail
        of the sheet from the new rollout."""
        token = self.aggregator.consume_token()
        self.engine.commit(token)
        self._tick_count += 1

        if self.engine.is_done:
            return

        # Replace the future-tail of the sheet with events derived from
        # the new rollout. Past events (sheet[:spawned_count]) are left
        # alone — they're already on screen as Figures.
        new_tail = tokens_to_events(self.engine.rollout(), self.col_start)
        self.sheet[self.spawned_count:] = new_tail

        if self.debug:
            self._dbg(token)

    # ------------------------------------------------------------------
    # Convenience read-throughs for the display
    # ------------------------------------------------------------------
    @property
    def last_action(self):
        """One of: 'init', 'match', 'silence', 'deviation', 'locked', 'done'."""
        return self.engine.last_action

    @property
    def current_pitch(self):
        """MIDI pitch the engine currently believes the player is on."""
        return self.engine.current_pitch

    # Visual lookahead between figure spawn and the play line, in ticks.
    # Used to map engine.current_tick to "what bar the player is reading".
    _PLAY_LINE_LOOKAHEAD = 15

    def _bar_at_play_line(self):
        """Index of the bar whose chord matches what's currently at the
        display's play line. The engine is ~15 ticks ahead of the player
        (visual lookahead), so we subtract that."""
        t = max(0, self.engine.current_tick - self._PLAY_LINE_LOOKAHEAD)
        return t // 16    # TICKS_PER_BAR = 16

    @property
    def current_chord(self):
        """Chord string at the bar currently being played at the play line."""
        bars = self.engine.bar_chords or []
        if not bars:
            return ""
        return bars[min(self._bar_at_play_line(), len(bars) - 1)]

    @property
    def upcoming_chord(self):
        """Chord string at the *next* bar after the play line."""
        bars = self.engine.bar_chords or []
        if not bars:
            return ""
        return bars[min(self._bar_at_play_line() + 1, len(bars) - 1)]

    # ------------------------------------------------------------------
    # Optional: convert what the player just played to a human-readable
    # token name. Useful for HUD overlays / debugging.
    # ------------------------------------------------------------------
    @staticmethod
    def token_label(token):
        t = int(token)
        if t == TOK_REST:
            return "REST"
        if t == TOK_HOLD:
            return "HOLD"
        return f"NOTE_{40 + t - 2}"

    def _dbg(self, token):
        head = self.sheet[self.spawned_count:self.spawned_count + 5]
        print(f"[t={self._tick_count:>4d}] "
              f"played={self.token_label(token):<8s} "
              f"action={self.last_action:<9s} "
              f"curr_pitch={self.current_pitch}  "
              f"sheet[{self.spawned_count}:{self.spawned_count + 5}]={head}")
