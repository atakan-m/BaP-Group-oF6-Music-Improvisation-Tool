"""Realtime glue between the JazzImprov engine and Piano_display.

`Piano_display` reads a `sheet = [[col, length], ...]` list and spawns one
figure per entry, waiting `length` ticks between spawns. This module owns
the engine, the mic aggregator, and the sheet — keeping it up to date
each 16th-note tick:

    1. Consume player input from the SP aggregator.
    2. Commit it to the engine (advances persistent state, possibly
       triggers a deviation rebuild of the rollout).
    3. If the engine just committed a NOTE_x, **append a new event** to
       the sheet with length = 1 + (count of leading HOLDs in the new
       rollout). HOLD/REST commits don't add events — they're already
       baked into the previous note's length.

This commit-driven incremental design has a few nice properties:

  * The sheet only ever **grows**, never gets truncated or retroactively
    rewritten — so Piano_display can never run out of upcoming events.
  * On a deviation, the engine commits the player's NOTE_y. The new
    rollout begins with the forced lock HOLDs, so the deviation note's
    length is exactly the lock duration (4 ticks = quarter). The player
    sees their wrong note rendered as a held quarter, then upcoming
    notes reflect the model's recovery plan.
  * Silence-as-match: when the player isn't playing, the engine commits
    its own predicted token. NOTE_x predictions appear in the sheet
    immediately, so the display keeps showing fresh material.

Wiring from Piano_display.py:

    from realtime_sheet import RealtimeSheet
    rs = RealtimeSheet(chord_prog, total_bars=64)
    sheet = rs.sheet                 # share the list reference

    # Hook the SP detector callback so every audio block feeds the aggregator
    _orig = detector.callback
    def _hooked(indata, frames, time_info, status):
        _orig(indata, frames, time_info, status)
        rs.aggregator.observe(detector.last_midi)
    detector.callback = _hooked

    game = Music(HEIGHT, WIDTH, sheet)
    rs.attach_music(game)

    # In the per-16th-note block of the main loop, BEFORE game.new_figure():
    if counter % (Testing_variable_for_testing) == 0:
        rs.tick()                    # ← commits first, may append to sheet
        game.new_figure()             # ← then spawn from updated sheet
        ...
"""

from engine import (
    make_engine, MicTickAggregator,
    TOK_REST, TOK_HOLD, TOK_NOTE_BASE,
    PITCH_LOW,
)


# ----------------------------------------------------------------------
# The realtime sheet controller
# ----------------------------------------------------------------------

class RealtimeSheet:
    """Drives a JazzImprov engine per 16th-note tick and exposes a
    Piano_display-compatible `sheet` of [col, length] events.

    State:
      * `self.engine`     — JazzImprov instance
      * `self.aggregator` — MicTickAggregator (hook the SP callback into it)
      * `self.sheet`      — the list Music reads; mutated in place (appended)
      * `self._music`     — attached Music (so we can read spawn progress)
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

        # Sheet is built INCREMENTALLY from committed tokens. Starts empty;
        # the first few engine commits (silence-as-match) populate it within
        # ~3–5 wall-clock ticks (max_consec_holds caps the gap between NOTEs).
        self.sheet = []

        self._music = None
        self._tick_count = 0
        # On deviation we pre-populate sheet[spawned_count:] with the
        # deviation note + the model's post-lock predictions, so Music
        # always has something to spawn during the lock period. The engine
        # will later commit those same predictions via silence-as-match and
        # the incremental append branch would duplicate them — this counter
        # tells it to skip the next N NOTE_x appends.
        self._suppress_appends = 0

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach_music(self, music):
        """Connect the Music instance so we know how many sheet events
        have been spawned (used for the debug HUD only — sheet itself is
        append-only and doesn't need this for correctness)."""
        self._music = music

    @property
    def spawned_count(self):
        """Number of sheet events Piano_display has already spawned."""
        return self._music.note if self._music is not None else 0

    # ------------------------------------------------------------------
    # The per-tick loop
    # ------------------------------------------------------------------
    def tick(self):
        """Call exactly once per 16th-note tick, BEFORE game.new_figure().

        Three cases:

        * **deviation** — the player just played a wrong note. Drop the
          stale upcoming plan (`sheet[spawned_count:]`) and replace it
          with the deviation note as a held quarter (length = 1 + the
          engine's forced lock HOLDs). The next ~3 ticks will be locked
          and won't touch the sheet; the post-lock NOTE commits will
          then incrementally fill in the model's new plan.
        * **match / silence** — the engine committed its plan token. If
          it was a NOTE_x, append a new event to the sheet with length
          = 1 + leading HOLDs in the rollout. HOLDs/RESTs are absorbed
          into the previous note's length and add nothing.
        * **locked / done / init** — leave the sheet alone.
        """
        token = self.aggregator.consume_token()
        self.engine.commit(token)
        self._tick_count += 1

        if self.engine.is_done:
            return

        action = self.engine.last_action

        # ------------- deviation: realtime re-plan -------------
        if action == "deviation":
            # Determine the deviation pitch:
            #   * If the player attacked a new NOTE_x → use that pitch.
            #   * If they sustained a HOLD while the model expected a new
            #     note → use engine.current_pitch (the pitch they're on).
            committed = int(self.engine.prev_actual)
            if committed >= TOK_NOTE_BASE:
                midi = PITCH_LOW + (committed - TOK_NOTE_BASE)
            else:
                midi = self.engine.current_pitch
            col = max(0, midi - self.col_start)

            # Length = 1 (the NOTE) + n_lock forced HOLDs at the front of
            # the freshly-rebuilt rollout (the engine's lock period).
            rollout = self.engine.rollout()
            n_lock = 0
            for tok in rollout:
                if int(tok) == TOK_HOLD:
                    n_lock += 1
                else:
                    break
            dev_event = [col, 1 + n_lock]

            # Extract the model's POST-lock predictions so Music has events
            # to spawn during the lock instead of staring at an empty queue.
            post_lock_events = self._tokens_to_events(rollout[n_lock:])

            # Drop the stale upcoming plan; insert the deviation note +
            # the new post-lock plan as the next things to scroll in.
            self.sheet[self.spawned_count:] = [dev_event] + post_lock_events

            # The engine will commit those same post-lock NOTEs over the
            # next ~rollout_ticks ticks via silence-as-match. Tell the
            # incremental append branch to skip exactly len(post_lock_events)
            # NOTE_x commits so we don't add the same events twice.
            self._suppress_appends = len(post_lock_events)

            if self.debug:
                print(f"  >>> DEVIATION midi={midi} col={col} "
                      f"length={1 + n_lock}  +{len(post_lock_events)} post-lock events  "
                      f"spawned_count={self.spawned_count}  "
                      f"sheet_len_now={len(self.sheet)}")

        # ------------- match / silence: incremental append -------------
        elif action in ("match", "silence"):
            committed = int(self.engine.prev_actual)
            if committed >= TOK_NOTE_BASE:
                if self._suppress_appends > 0:
                    # This NOTE was already inserted into the sheet by the
                    # deviation pre-population; don't double-add it.
                    self._suppress_appends -= 1
                else:
                    midi = PITCH_LOW + (committed - TOK_NOTE_BASE)
                    col  = max(0, midi - self.col_start)
                    length = 1
                    for tok in self.engine.rollout():
                        if int(tok) == TOK_HOLD:
                            length += 1
                        else:
                            break
                    self.sheet.append([col, length])

        # ------------- locked / done / init: sheet stays put -------------

        if self.debug:
            self._dbg(token)

    # ------------------------------------------------------------------
    # Helper: token stream → discrete [col, length] events
    # ------------------------------------------------------------------
    def _tokens_to_events(self, tokens):
        """Walk a token sequence and emit a list of [col, length] events.
        Leading HOLDs / RESTs are skipped; each NOTE_x starts a new event
        whose length is 1 + (count of immediately-following HOLDs)."""
        events = []
        i, n = 0, len(tokens)
        while i < n:
            tok = int(tokens[i])
            if tok >= TOK_NOTE_BASE:
                midi = PITCH_LOW + (tok - TOK_NOTE_BASE)
                col = max(0, midi - self.col_start)
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
    # Debug helpers
    # ------------------------------------------------------------------
    @staticmethod
    def token_label(token):
        t = int(token)
        if t == TOK_REST:
            return "REST"
        if t == TOK_HOLD:
            return "HOLD"
        return f"NOTE_{40 + t - TOK_NOTE_BASE}"

    def _dbg(self, token):
        head = self.sheet[self.spawned_count:self.spawned_count + 5]
        print(f"[t={self._tick_count:>4d}] "
              f"played={self.token_label(token):<8s} "
              f"action={self.last_action:<9s} "
              f"curr_pitch={self.current_pitch}  "
              f"sheet_len={len(self.sheet)}  "
              f"upcoming[{self.spawned_count}:+5]={head}")
