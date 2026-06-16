"""Main application: realtime jazz-improvisation tool with pygame display.

Pulls a chord progression from Buttons_v2, builds the realtime engine,
opens the microphone stream, and runs the per-frame display loop. Every
16th-note boundary the engine commits the player's input and the on-screen
figures are repainted from the new rollout; every frame the figures scroll
down by `speed` y-units.
"""

import pygame
import random
import numpy as np
from collections import deque
import sys
sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//ML')
sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//SP')
sys.path.insert(1, '..//BaP-Group-oF6-Music-Improvisation-Tool//HW//bap')
import generate
import torch
import json
import signalprocessing             # SP/signalprocessing.py — pitch detector
import Buttons_v2
import sounddevice as sd
from perf import PerfLogger          # ML/perf.py — realtime timing buckets

# Audio / timing constants pulled from the SP detector.
SAMPLE_RATE = signalprocessing.SAMPLE_RATE     # 44100
HOP_SIZE    = signalprocessing.HOP_SIZE        # 512 samples ≈ 11.6 ms / callback

# Display window dimensions (in "tick" units; one tick is one 16th note).
WIDTH  = 36
HEIGHT = 20

# Framerate and tempo. Testing_variable_for_testing = frames per 16th note;
# speed is how far a figure scrolls each frame, sized so it covers one
# tick of vertical space per 16th-note window.
fps = 60
bpm = 60
Testing_variable_for_testing = (fps*fps//bpm//4)
speed = 1/Testing_variable_for_testing

# Pitch detector — exposes detector.last_midi (int MIDI or None for silence)
# on every audio block. We wrap its callback below to also feed the mic
# aggregator that buffers detections inside each 16th-note window.
detector = signalprocessing.PitchDetector()

# Horizontal positions (in zoom-units) of the 37 piano-key slots used to
# draw both the falling note rectangles and the keyboard at the bottom.
# key_width is the relative width of each key (1.0 = white key, 0.5 = black).
hor_pos = [0,0.5,1,1.5,2,3,3.5,4,4.5,5,5.5,6,7,7.5,8,8.5,9,10,10.5,11,11.5,12,12.5,13,14,14.5,15,15.5,16,17,17.5,18,18.5,19,19.5,20,-10]
key_width = [1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1]

# 16-row pre-roll buffer (15 silent + 1 anchor row at index 15). This length
# is chosen so engine_tick 0 reaches the play line at *exactly* tally 31 —
# the same tally the engine does its first commit — so what the player sees
# at the play line is what the engine is committing. Don't change without
# also updating `_PLAY_LINE_OFFSET` and the engine-commit gate (`tally >= 31`).
# Each "row" is the column index of the single note at that tick (None for
# silence; we're monophonic so one int per tick is enough).
buffer_sheet = [None] * 15 + [17]

col_start = 48                       # MIDI 48 (C3) is column 0 on the display


def midi_to_col(midi):
    """Map a MIDI pitch to a display column 0..35, or None if out of range."""
    col = midi - col_start
    return col if 0 <= col < 36 else None


# Note name for every visible column (one entry per semitone, 0..35).
Note_list = [
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
]


# Set True to print one line per 16th-note tick with the engine's commit
# action and timing. Useful when developing on a workstation; keep False
# on the Raspberry Pi to avoid stdout overhead inside the inner loop.
DEBUG_REALTIME = True

# ─────────────────────── Realtime perf logging ────────────────────────
# Two PerfLoggers measure where the per-tick time goes:
#   perf_commit     — wraps just engine.commit() (LSTM forwards + engine
#                     bookkeeping; bucketed by engine.last_action).
#   perf_tick_block — wraps the entire main-thread per-tick block (commit
#                     + render_tokens + on-screen figure update + sheet
#                     update + spawn + cleanup).
# Deviation commits are the spike: they rebuild the rollout and the per-
# tick LSTM count goes up. On the Pi those are the calls that risk drift
# if performance is tight, so they're the numbers worth watching.
#
# Tunables:
#   PERF_PRINT_EVERY_N_TICKS — terminal summary cadence (0 disables).
#   PERF_RECENT_WINDOW       — number of recent samples per bucket
#                              included in periodic prints. Lets you see
#                              what perf looks like *now*, not the
#                              all-time average.
#   PERF_CSV_ON_QUIT         — filename to dump every raw sample to on
#                              exit (set to None to skip).
PERF_PRINT_EVERY_N_TICKS = 100
PERF_RECENT_WINDOW       = 100
PERF_CSV_ON_QUIT         = "commit_timings.csv"

perf_commit     = PerfLogger("engine.commit")
perf_tick_block = PerfLogger("full_tick_block")

# ─────────────────────── Display rendering helpers ────────────────────
# Per-tick column rendering is monophonic — each tick maps to one column
# index (or None for silence). REST/HOLD reuse the previous pitch's column
# because a 1-pixel "rest" marker isn't worth drawing. The render function
# is stateless: you pass the previous pitch in and receive an updated one
# back, so the future of the rollout can be re-rendered on every commit
# without touching any global state.
PITCH_LOW_DISPLAY     = 48              # MIDI 48 = C3
PITCH_HIGH_DISPLAY    = 83              # MIDI 83 = B5
DEFAULT_DISPLAY_PITCH = 60


# ─────────────────────── Mic → tick aggregator ────────────────────────
# The SP callback fires every ~11.6 ms on the sounddevice audio thread.
# The aggregator buffers all detections inside the current 16th-note
# window and emits one REST/HOLD/NOTE token at the tick boundary. We wrap
# the detector's callback so each audio block also feeds the aggregator
# before returning. The wrap must be installed *before* the InputStream
# is started further down so it captures the wrapped callable.
mic_aggregator = generate.MicTickAggregator()

_original_detector_callback = detector.callback
def _detector_callback_with_aggregation(indata, frames, time_info, status):
    _original_detector_callback(indata, frames, time_info, status)
    mic_aggregator.observe(detector.last_midi)
detector.callback = _detector_callback_with_aggregation

class Figure:
    """One on-screen note rectangle representing a single 16th-note tick.

    Stores the column index (or None for silence) plus the wall-clock tally
    at spawn. The source_idx lets the realtime engine find this figure later
    and overwrite its column after a re-plan; without it only freshly-spawned
    figures would visibly react to deviations.

    The rendering pass groups consecutive same-column figures into one tall
    rectangle, so a sustained 4-tick note shows as a single block — the
    same visual as a discrete-event design, but with the per-tick internal
    model the realtime engine needs.
    """

    def __init__(self, col, source_idx=None):
        self.x = 0
        self.y = 0
        self.source_idx = source_idx
        self._col_idx = col
        self._note_name = Note_list[col] if col is not None else ""

    @property
    def col_idx(self):
        """Column index 0..35, or None for a silent tick."""
        return self._col_idx

    @property
    def note_name(self):
        """The displayed note name (e.g. 'C', 'F#'); empty for silent ticks."""
        return self._note_name

    def update_image(self, col):
        """Replace this figure's column in response to a model re-plan.

        Called by the realtime engine after a deviation rebuilds the rollout
        so figures already scrolling toward the play line visibly reflect
        the new plan. This is the single line responsible for making
        deviations visible *near* the play line, not only at the top of the
        screen.
        """
        self._col_idx = col
        self._note_name = Note_list[col] if col is not None else ""


class BeatBar:
    """A horizontal bar-divider line scrolling with the rest of the display.

    Each line optionally carries a chord label that is rendered at the
    left-edge of the playing area; the label scrolls down with the line.
    Labels are only set when the new bar's chord differs from the previous
    bar's, so they announce upcoming chord changes.
    """

    def __init__(self, label=""):
        self.y = 0
        self.label = label


class Music:
    """Holds the on-screen state: a deque of figures and a list of bar lines."""

    def __init__(self, height, width):
        self.x = 40
        self.y = 40
        self.zoom = 50
        self.figure = deque()
        # The bar-lines list starts empty. The main loop's per-bar block
        # spawns the first one at tally=16, which reaches the play line at
        # tally=31 — exactly when engine_tick 0 starts being played.
        self.beat_bars = []
        self.height = height
        self.width = width

    def new_figure(self, col, source_idx=None):
        """Append a new figure with the given column and spawn tally."""
        self.figure.append(Figure(col, source_idx=source_idx))

    def new_beatbar(self, label=""):
        """Spawn a new bar-divider line.

        If `label` is non-empty it's rendered to the left of the line as it
        descends, used to announce an upcoming chord change. The list is
        capped at three concurrent lines.
        """
        self.beat_bars.append(BeatBar(label=label))
        if len(self.beat_bars) > 3:
            self.beat_bars.pop(0)

    def go_down(self):
        """Advance every figure and bar line by one frame's worth of speed."""
        for k in range(len(self.figure)):
            self.figure[k].y += speed
        for bar in self.beat_bars:
            bar.y += speed



def render_tokens(tokens, start_prev_midi):
    """Convert a token sequence into per-tick column indices.

    REST and HOLD tokens reuse the previously-active pitch (because the
    monophonic display can't render an explicit rest glyph). NOTE_x tokens
    update the active pitch, clamped to the display's MIDI range. The
    function is stateless apart from the caller-supplied `start_prev_midi`:
    the returned `final_prev_midi` can be fed back in to continue the
    chain across multiple calls.
    """
    cols = []
    prev = start_prev_midi
    for tok in tokens:
        tok = int(tok)
        if tok >= 2:                                # NOTE_<midi>
            midi = 40 + (tok - 2)
            if midi < PITCH_LOW_DISPLAY:  midi = PITCH_LOW_DISPLAY
            if midi > PITCH_HIGH_DISPLAY: midi = PITCH_HIGH_DISPLAY
            prev = midi
        cols.append(midi_to_col(prev))
    return cols, prev


def get_played_token():
    """Consume the mic aggregator's window and return one token for this tick."""
    return mic_aggregator.consume_token()


# ─────────────────────── Chord display helpers ────────────────────────
# At tally T the figure at the play line has source_idx = T - 15 (it spent
# 15 ticks scrolling from y=0 to y=15). With the 16-row pre-roll buffer,
# total_sheet[S] represents engine_tick (S - 16), so the engine_tick at
# the play line is (T - 15) - 16 = T - 31. This matches what the engine
# commits at tally T (it advances current_tick from T-31 to T-30), so
# play-line content and engine commit are aligned tick-for-tick.
_PLAY_LINE_OFFSET = 31


def chord_at_play_line(tally):
    """Return the chord string currently sitting at the play line.

    Empty string before any chord progression is set; clamps to the last
    bar once the song has run past its total length.
    """
    bars = engine.bar_chords or []
    if not bars:
        return ""
    eng_tick = max(0, tally - _PLAY_LINE_OFFSET)
    return bars[min(eng_tick // 16, len(bars) - 1)]


def chord_upcoming(tally):
    """Return the chord of the next bar after the one at the play line."""
    bars = engine.bar_chords or []
    if not bars:
        return ""
    eng_tick = max(0, tally - _PLAY_LINE_OFFSET)
    return bars[min(eng_tick // 16 + 1, len(bars) - 1)]


# ────────────────────────── Pygame setup ──────────────────────────────
# Per-column x-positions in pixels for the 37 piano-key slots.
real_pos = [40 + 50 * hor_pos[p] * 36/21 + 2 for p in range(37)]

def run_game(total_sheet):
    """Boot pygame, open the mic stream, and run the main display loop."""
    pygame.init()
    print(" starting ")

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY  = (128, 128, 128)
    RED   = (255, 0, 0)

    size = (1920, 1080)
    screen = pygame.display.set_mode(size, pygame.FULLSCREEN, vsync=1)
    font     = pygame.font.SysFont('Calibri', 40, True, False)
    font1    = pygame.font.SysFont('Calibri', 50, True, False)
    font_bar = pygame.font.SysFont('Calibri', 24, True, False)   # bar-line chord labels
    pygame.display.set_caption("Jazz")

    done = False
    clock = pygame.time.Clock()

    game = Music(HEIGHT, WIDTH)
    counter = 0           # increments every frame; drives all timing
    tally   = 0           # 16th-note tick counter; matches engine.current_tick + 31

    background = pygame.Surface(size)
    background.fill(WHITE)

    # ── Background: the piano-key visual at the bottom + bar dividers ────
    for j in range(21 + 1):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 36/21, game.y + game.zoom * 15, 1, game.zoom * 5], 1)
    for j in range(4):
        pygame.draw.rect(background, BLACK, [game.x + 5.15 * game.zoom + game.zoom * j * 12, game.y , 1, game.zoom * 15], 1)
    for j in range(4):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 12, game.y, 1, game.zoom * 15], 1)
    for j in range(3):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 12 + game.zoom * 1.4 - 3, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.9,  game.zoom* 3], 100)
    for j in range(3):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 12 + game.zoom * 3.25 - 3, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.9,  game.zoom* 3], 100)
    for j in range(3):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 12 + game.zoom * 6.55 - 3, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.9,  game.zoom* 3], 100)
    for j in range(3):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 12 + game.zoom * 8.33 - 3, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.9,  game.zoom* 3], 100)
    for j in range(3):
        pygame.draw.rect(background, BLACK, [game.x + game.zoom * j * 12 + game.zoom * 10.11 - 3, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.9, game.zoom* 3], 100)
    pygame.draw.line(background, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * 0],  [game.x + game.zoom * WIDTH, game.y + game.zoom * 0])
    pygame.draw.line(background, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * 20], [game.x + game.zoom * WIDTH, game.y + game.zoom * 20])
    pygame.draw.line(background, RED,  [game.x + game.zoom * 0, game.y + game.zoom * 15], [game.x + game.zoom * WIDTH, game.y + game.zoom * 15])

    sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                blocksize=HOP_SIZE, dtype="float32",
                callback=detector.callback).start()


    # ────────────────────────────── Main loop ─────────────────────────────
    while not done:
        counter += 1
        if counter > 100000:
            counter = 0

        if counter % (Testing_variable_for_testing) == 0:
            # ──────────────────────────────────────────────────────────────
            # PER-TICK REALTIME BLOCK — runs once per 16th-note boundary.
            #
            # The display has 31 ticks of visual lookahead (16 pre-roll
            # rows + 15 ticks of scroll from spawn at y=0 to the play line
            # at y=15). The player can only play the model's first
            # prediction at wall-clock tally = 31. Before that we must
            # NOT commit to the engine, otherwise it would receive 30
            # ticks of bogus silence-as-match commits and bias itself into
            # long, dull sustains.
            #
            # After tally >= 31 each tick does:
            #   1. read the dominant pitch from the SP aggregator over
            #      the last 16th window;
            #   2. commit that token to the engine;
            #   3. overwrite total_sheet[tally..] from rollout[14..] so
            #      upcoming spawns reflect the new plan;
            #   4. update_image on existing on-screen figures from
            #      rollout[0..13] — this is what makes deviations visible
            #      near the play line, not only at the top of the screen.
            # ──────────────────────────────────────────────────────────────
            # Sentinel — set only if we actually start the tick-block
            # timer. `tally` is incremented further down inside this
            # branch, so we can't rely on the same `tally >= 31` test at
            # the closing perf_tick_block.stop call.
            _t_block = None
            if not engine.is_done and tally >= 31:
                # Wrap the entire tick block so we can compare the cost of
                # engine.commit against everything else (render + figure
                # updates + sheet extension + spawn).
                _t_block = perf_tick_block.start()

                played_token = get_played_token()

                # The expensive call — bucket by what kind of commit it ended
                # ended up being. Cost regimes (with default knobs):
                #   match / silence : ~2 LSTM forwards (advance + extend)
                #   locked          : ~2 forwards (advance + extend)
                #   deviation       : ~5–6 forwards (advance + n_lock
                #                     simulation + deviation_initial_fresh
                #                     samples)
                _t_commit = perf_commit.start()
                rollout = engine.commit(played_token)
                commit_ms = perf_commit.stop(_t_commit, engine.last_action)

                if DEBUG_REALTIME:
                    if played_token == 0:
                        played_str = "REST"
                    elif played_token == 1:
                        played_str = "HOLD"
                    else:
                        played_str = f"NOTE_{40 + played_token - 2}"
                    buf_preview = list(engine._rollout_buf[:5])
                    print(f"[t={tally:>4d}] played={played_str:<8s} "
                        f"curr_pitch={engine.current_pitch}  "
                        f"action={engine.last_action:<9s}  "
                        f"commit={commit_ms:>6.1f}ms  "
                        f"buf[0:5]={buf_preview}")
                if rollout:
                    # Render the full rollout in one pass so the prev-pitch
                    # chain across HOLD/REST tokens is consistent. We seed
                    # it with engine.current_pitch — which after a deviation
                    # is the deviation note, not the stale pitch from before.
                    spawn_prev = (engine.current_pitch
                                if engine.current_pitch is not None
                                else DEFAULT_DISPLAY_PITCH)
                    rollout_cols, _ = render_tokens(rollout, spawn_prev)

                    # (a) Update on-screen figures from rollout[0..13].
                    # A figure with source_idx S is at y = tally - 1 - S
                    # right now; its rollout slot is k = S - tally + 14,
                    # valid only for 0 <= k < 14. This is what makes the
                    # next note at the bottom of the display visibly react
                    # to a deviation, not only the figures still to spawn.
                    for fig in game.figure:
                        if fig.source_idx is None:
                            continue
                        k = fig.source_idx - tally + 14
                        if 0 <= k < 14 and k < len(rollout_cols):
                            fig.update_image(rollout_cols[k])

                    # (b) Overwrite (or extend) total_sheet from
                    # rollout[14:] so upcoming spawns reflect the new plan.
                    for offs, col in enumerate(rollout_cols[14:]):
                        idx = tally + offs
                        if idx < len(total_sheet):
                            total_sheet[idx] = col
                        else:
                            total_sheet.append(col)

            # Spawn the next figure at the top of the screen.
            if tally < len(total_sheet):
                game.new_figure(total_sheet[tally], source_idx=tally)
                tally += 1
            # Pop the oldest figure once it has scrolled past the keyboard
            # visualisation, or if it represents silence (no column).
            if len(game.figure) != 0 and game.figure[0].y > 16:
                game.figure.popleft()
            if len(game.figure) != 0 and game.figure[0].col_idx is None:
                game.figure.popleft()

            # Close out the whole-tick timer. The sentinel — rather than
            # the `tally >= 31` test — gates this because `tally` may have
            # just been incremented inside this branch.
            if _t_block is not None:
                perf_tick_block.stop(_t_block, engine.last_action)

            # Periodic perf summary. Useful on the Pi to confirm that the
            # deviation spikes are still inside the 16th-note budget
            # (≈107 ms at 140 BPM; ≈250 ms at 60 BPM).
            if (PERF_PRINT_EVERY_N_TICKS
                    and tally >= 31
                    and tally % PERF_PRINT_EVERY_N_TICKS == 0):
                tick_budget_ms = 60_000.0 / bpm / 4
                print(f"\n  16th-note budget at {bpm} BPM = {tick_budget_ms:.1f} ms")
                perf_commit.print_summary(recent_n=PERF_RECENT_WINDOW)
                perf_tick_block.print_summary(recent_n=PERF_RECENT_WINDOW,
                                            header=False)

        if counter % (Testing_variable_for_testing * 16) == 0:
            # A bar line spawned now reaches the play line in 15 ticks.
            # With the 16-row pre-roll buffer that lines up with
            # engine_tick = tally − 16 crossing the play line — i.e. the
            # start of bar (tally − 16) // 16. We label the line with that
            # bar's chord, but only if it differs from the previous bar so
            # the label exclusively announces real chord *changes*.
            bars = engine.bar_chords or []
            bar_idx = (tally - 16) // 16
            if 0 <= bar_idx < len(bars):
                new_chord = bars[bar_idx]
                prev_chord = bars[bar_idx - 1] if bar_idx > 0 else None
                chord_label = new_chord if new_chord != prev_chord else ""
            else:
                chord_label = ""
            game.new_beatbar(chord_label)

        game.go_down()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.__init__(HEIGHT, WIDTH)
                if event.key == pygame.K_ESCAPE:
                    done = True
                if event.key == pygame.K_q:
                    return

        screen.blit(background, (0, 0))

        # ── Figures: render the falling notes. Consecutive same-column
        #    figures are merged into one tall rectangle, so a sustained note
        #    appears as a single variable-length block on screen — but the
        #    per-tick internal model is preserved underneath, which is what
        #    allows the realtime engine to update figures already in flight
        #    after a deviation (see the update_image loop above). ──
        if game.figure is not None and len(game.figure) > 0:
            figs = list(game.figure)
            i = 0
            while i < len(figs):
                fig = figs[i]
                col = fig.col_idx
                if col is None:
                    i += 1
                    continue
                # Find the run of consecutive same-column figures from i.
                j = i + 1
                while j < len(figs) and figs[j].col_idx == col:
                    j += 1
                # Group spans figs[i..j-1]. figs[0] is oldest (highest y),
                # figs[-1] newest (lowest y); the last in the group is the
                # newest one, i.e. the top of the visual rectangle.
                top_fig = figs[j - 1]
                n_ticks = j - i
                if top_fig.y > 22 or (top_fig.y + n_ticks - 1) < -2:
                    i = j
                    continue                # entirely off-screen
                pygame.draw.rect(
                    screen, (0, 100 * key_width[col], 0),
                    [real_pos[col] + game.zoom * (0.5 / key_width[col] - 0.5),
                    game.y + game.zoom * (top_fig.y - 1) + 1,
                    1.7 * game.zoom * key_width[col] - 3,
                    game.zoom * n_ticks - 3],
                )
                # Note name at the top of the rectangle.
                screen.blit(
                    font.render(top_fig.note_name, True, RED),
                    [real_pos[col] + (game.zoom * 0.5),
                    game.y + game.zoom * (top_fig.y - 1)],
                )
                i = j

        # Bar lines + their chord-change labels. Labels are rendered at the
        # left edge of the playing area just above the line so they scroll
        # down with the line without overlapping it.
        if game.beat_bars:
            for bar in game.beat_bars:
                if bar.y > 15 or bar.y < -2:
                    continue
                line_py = game.y + game.zoom * bar.y - 3
                pygame.draw.line(screen, GRAY,
                                [game.x + game.zoom * 0,     line_py],
                                [game.x + game.zoom * WIDTH, line_py])
                if bar.label:
                    lbl = font_bar.render(bar.label, True, BLACK)
                    screen.blit(lbl, [game.x + 2, line_py - lbl.get_height()])

        # ── Top-of-screen HUD: BPM, elapsed time, currently-detected note,
        #    current chord, next chord. The "Cur note" readout doubles as a
        #    blue marker on the bottom keyboard so the player gets visual
        #    feedback on what the mic is hearing. ──
        text1 = font1.render("BPM: " + str(bpm), True, BLACK)
        text2 = font1.render("Time: " + str(counter // fps), True, BLACK)
        if detector.last_midi is not None:
            _m = detector.last_midi
            _name = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][_m % 12] + str(_m // 12 - 1)
            text3 = font1.render("Cur note: " + _name, True, BLACK)
            # Blue marker on the keyboard at the bottom.
            pygame.draw.rect(
                screen, (0, 0, 255),
                [real_pos[(_m - 48) % 37] + game.zoom * (0.5 / key_width[(_m - 48) % 37] - 0.5),
                game.zoom * 17,
                1.7 * key_width[(_m - 48) % 37] * game.zoom - 3,
                game.zoom * 2],
            )
        else:
            text3 = font1.render("Cur note: ", True, BLACK)

        text4 = font1.render("Chord: "      + chord_at_play_line(tally), True, BLACK)
        text5 = font1.render("Next chord: " + chord_upcoming(tally),     True, BLACK)

        screen.blit(text1, [0, 0])
        screen.blit(text2, [game.zoom * 6, 0])
        screen.blit(text3, [game.zoom * 12, 0])
        screen.blit(text4, [game.zoom * 20, 0])
        screen.blit(text5, [game.zoom * 28, 0])

        pygame.display.flip()
        clock.tick(fps)
        
    # ───────────────────── Perf summary + CSV on quit ─────────────────────
    # Print the final stats table and optionally write the raw samples and
    # the aggregated bucket summary to CSV files for offline analysis.
    print("\n" + "=" * 60)
    print("FINAL PERF SUMMARY")
    print("=" * 60)
    print(f"16th-note budget at {bpm} BPM = {60_000.0 / bpm / 4:.1f} ms")
    perf_commit.print_summary()
    perf_tick_block.print_summary(header=False)

    if PERF_CSV_ON_QUIT:
        perf_commit.save_csv(PERF_CSV_ON_QUIT)
        summary_path = PERF_CSV_ON_QUIT.replace(".csv", "_summary.csv")
        perf_commit.save_summary_csv(summary_path)
        print(f"\nWrote raw samples → {PERF_CSV_ON_QUIT}")
        print(f"Wrote bucket summary → {summary_path}")



    pygame.quit()
    return

# ─────────────────────── Realtime engine setup ────────────────────────
# Top-level driver loop. Each iteration:
#   1. asks the player to enter a chord progression via Buttons_v2;
#   2. builds a fresh realtime engine for that progression;
#   3. pre-fills the on-screen sheet from the engine's initial rollout so
#      the player has notes to read during the 31-tick pre-roll period;
#   4. enters the pygame display loop (run_game) until the user quits.
# Quitting run_game (Q key) returns here, so a new song can be entered
# without restarting the program.
while True:
    chord_prog_2 = []
    for chord in Buttons_v2.main():
        chord_prog_2.append((chord, 1))    # one bar per chord; engine cycles to fill
    engine = generate.make_realtime_engine(
        chord_prog_2,

        total_bars=16,
        # ↓↓↓  PER-RUN TUNING (the checkpoint file used at inference is the
        #      one named in engine.py's DEFAULT_CHECKPOINT)             ↓↓↓
        temperature=0.9,             # 0.4 ballad, 0.8 bebop
        rollout_ticks=30,            # ~5.6 s of lookahead — covers the display window
        deviation_lock_ticks=4,      # quarter-note lock after a deviation
        hold_penalty=1.0,            # positive → shorter notes, negative → longer
        rest_penalty=1.5,            # bias against silence so the model keeps playing
        max_consec_holds=4,          # hard cap — notes never exceed a quarter
        # ── DEVIATION COST AMORTISATION ───────────────────────────────────
        # Without amortisation a deviation tick did ~rollout_ticks LSTM
        # forwards in one go, producing a visible CPU spike in the perf
        # log. The two knobs below spread that work across multiple ticks:
        #
        #   deviation_initial_fresh : tokens sampled fresh on the
        #                             deviation tick itself. With the
        #                             default n_lock = 3, a value of 2
        #                             gives roughly 1 advance + 3 lock
        #                             simulation + 2 sample = ~6 LSTM
        #                             forwards on the deviation tick.
        #
        #   rollout_extend_per_tick : extra tokens sampled per subsequent
        #                             commit while the rollout is still
        #                             below `rollout_ticks` length. With
        #                             4, a catch-up commit costs about
        #                             ~2 normal + 4 catch-up = ~6
        #                             forwards, and the rollout refills
        #                             in roughly 6 ticks.
        #
        # The visual trade-off is that immediately after a deviation only
        # the figures closest to the play line carry the new plan; the
        # change "wave" then propagates outward across the next few ticks
        # as the rollout regrows.
        deviation_initial_fresh=1,
        rollout_extend_per_tick=1,
    )

    # Pre-fill the displayable sheet with the engine's initial rollout so
    # the player has notes to read during the pre-roll period.
    initial_rollout = engine.rollout()
    initial_cols, _prev_chain_midi = render_tokens(initial_rollout, DEFAULT_DISPLAY_PITCH)
    total_sheet = buffer_sheet + initial_cols    # list of int|None, one per tick
    run_game(total_sheet)

