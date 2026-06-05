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
import signalprocessing             # SP/ — pitch detector
import Buttons_v2
import sounddevice as sd

SAMPLE_RATE = signalprocessing.SAMPLE_RATE     # 44100
HOP_SIZE    = signalprocessing.HOP_SIZE        # 512 = ~11.6 ms/callback
WIDTH = 36
HEIGHT = 20
fps = 60
bpm = 60
Testing_variable_for_testing = (fps*fps//bpm//4)
speed = 1/Testing_variable_for_testing

# Pitch detector — exposes `detector.last_midi` (int or None) every audio block.
detector = signalprocessing.PitchDetector()

hor_pos = [0,0.5,1,1.5,2,3,3.5,4,4.5,5,5.5,6,7,7.5,8,8.5,9,10,10.5,11,11.5,12,12.5,13,14,14.5,15,15.5,16,17,17.5,18,18.5,19,19.5,20,-10]
key_width = [1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1]

# 17-row pre-roll buffer: 16 silent + 1 anchor so the play line settles before
# the engine starts committing.
buffer_sheet = [
    [0]*36, [0]*36, [0]*36, [0]*36, [0]*36, [0]*36, [0]*36, [0]*36,
    [0]*36, [0]*36, [0]*36, [0]*36, [0]*36, [0]*36, [0]*36, [0]*36,
    [1 if c == 17 else 0 for c in range(36)],
]

col_start = 48                       # MIDI 48 = C3 → column 0


def note_to_col(note):
    return note - col_start


def notes_in_row(notes, num_cols=36):
    """A single 16th-note display row. `notes` is a tuple/list with one
    pitch; we set the bit at that pitch's column.

    col_start=48 → col=0 means MIDI 48 (C3). Use row[col] (not row[col-1])
    so the dot aligns with the C-labeled visual column at position 0 —
    the old `row[col-1]` was off by a semitone.
    """
    row = [0]*num_cols
    col = note_to_col(notes[0])
    if 0 <= col < num_cols:
        row[col] = 1
    return row


def make_sheet(notes, num_cols=36):
    return [notes_in_row(note, num_cols) for note in notes]


# 36 column rendering positions — one per semitone column 0..35.
Note_list = [
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
]


def _row_to_col(row):
    """Find the column index of the (single) set bit in a display row,
    or None if the row is silent."""
    for i, v in enumerate(row):
        if v:
            return i
    return None


class Figure:
    """One Figure per 16th-note tick. Carries the tick index it was spawned
    from so the realtime engine can find and re-render it after a deviation.

    On screen the per-tick figures get visually grouped into variable-length
    rectangles by the rendering pass at the bottom of the main loop, so a
    sustained 4-tick note shows as one tall block — same look as the
    discrete-event design, but with the per-tick model that realtime needs.
    """

    def __init__(self, Notes, source_idx=None):
        self.x = 0
        self.y = 0
        # Tick index this figure was spawned from. After a model re-plan
        # we use this to figure out which rollout slot replaces this
        # figure's image — without it, only future spawns would change.
        self.source_idx = source_idx
        self.tune = Notes
        self._col_idx = _row_to_col(Notes)
        self._note_name = Note_list[self._col_idx] if self._col_idx is not None else ""

    @property
    def col_idx(self):
        """0..35, or None for a silent row."""
        return self._col_idx

    @property
    def note_name(self):
        return self._note_name

    def update_image(self, Notes):
        """Re-render this figure from a new row. Called after the engine
        re-plans on deviation so figures already scrolling toward the play
        line visibly reflect the new plan instead of the old one — this is
        the single line that makes realtime *visible* near the play line."""
        self.tune = Notes
        self._col_idx = _row_to_col(Notes)
        self._note_name = Note_list[self._col_idx] if self._col_idx is not None else ""


class Music:
    def __init__(self, height, width):
        self.x = 40
        self.y = 40
        self.zoom = 33.333333
        self.figure = deque()
        self.beat_bars = [0]
        self.height = height
        self.width = width

    def new_figure(self, bar, source_idx=None):
        self.figure.append(Figure(bar, source_idx=source_idx))

    def new_beatbar(self):
        self.beat_bars.append(0)
        if len(self.beat_bars) > 3:
            self.beat_bars.pop(0)

    def go_down(self):
        for k in range(len(self.figure)):
            self.figure[k].y += speed
        for h in range(len(self.beat_bars)):
            self.beat_bars[h] += speed


# ─────────────────────── Realtime engine setup ────────────────────────
chord_prog_2 = []
for chord in Buttons_v2.main():
    chord_prog_2.append((chord, 1))    # 1 bar per chord; engine cycles to fill

engine = generate.make_realtime_engine(
    chord_prog_2,
    total_bars=64,
    # ↓↓↓  PER-RUN TUNING (model file lives at ML/models/jazz_lstm.pt;
    #      to switch models, edit DEFAULT_CHECKPOINT in ML/engine.py)  ↓↓↓
    temperature=0.45,            # 0.4 ballad, 0.8 bebop
    rollout_ticks=30,            # ~5.6s of lookahead — covers display window
    deviation_lock_ticks=4,      # quarter-note lock after a deviation
    hold_penalty=0.7,            # +ve → shorter notes, −ve → longer
    rest_penalty=1.5,            # suppress silence
    max_consec_holds=4,          # hard cap → notes never exceed a quarter
)

# Set True for per-tick commit + match/deviation logs in the terminal.
DEBUG_REALTIME = True

# ─────────────────────── Display rendering helpers ────────────────────
# Token sequence → 36-col display rows. REST/HOLD reuse the previous pitch
# (because notes_in_row always sets one bit, there's no way to draw rest).
# Stateless: pass the prev pitch in and back out so we can re-render the
# future on every commit without corrupting a global.
PITCH_LOW_DISPLAY  = 48              # MIDI 48 = C3
PITCH_HIGH_DISPLAY = 83              # MIDI 83 = B5
DEFAULT_DISPLAY_PITCH = 60


def render_tokens(tokens, start_prev_midi):
    """Render a token stream into display rows.
    Returns (rows, final_prev_midi)."""
    rows = []
    prev = start_prev_midi
    for tok in tokens:
        tok = int(tok)
        if tok >= 2:                                # NOTE_<midi>
            midi = 40 + (tok - 2)
            if midi < PITCH_LOW_DISPLAY:  midi = PITCH_LOW_DISPLAY
            if midi > PITCH_HIGH_DISPLAY: midi = PITCH_HIGH_DISPLAY
            prev = midi
        rows.append(notes_in_row((prev,), 36))
    return rows, prev


# Pre-fill the displayable sheet with the engine's initial rollout so the
# player has notes to read during the pre-roll period.
initial_rollout = engine.rollout()
sheet, _prev_chain_midi = render_tokens(initial_rollout, DEFAULT_DISPLAY_PITCH)
total_sheet = buffer_sheet + sheet


# ─────────────────────── Mic → tick aggregator ────────────────────────
# The SP callback fires every ~11.6ms; aggregate all detections inside the
# current 16th-note window and emit one REST/HOLD/NOTE token at the boundary.
mic_aggregator = generate.MicTickAggregator()

_original_detector_callback = detector.callback
def _detector_callback_with_aggregation(indata, frames, time_info, status):
    _original_detector_callback(indata, frames, time_info, status)
    mic_aggregator.observe(detector.last_midi)
detector.callback = _detector_callback_with_aggregation


def get_played_token():
    return mic_aggregator.consume_token()


# ─────────────────────── Chord display helpers ────────────────────────
# Player at the play line is reading the figure with source_idx = tally-16.
# Engine ticks start at total_sheet index 17 (after the 17-row buffer), so
# the engine tick at the play line is (tally - 16) - 17 = tally - 33.
_PLAY_LINE_OFFSET = 33


def chord_at_play_line(tally):
    bars = engine.bar_chords or []
    if not bars:
        return ""
    eng_tick = max(0, tally - _PLAY_LINE_OFFSET)
    return bars[min(eng_tick // 16, len(bars) - 1)]


def chord_upcoming(tally):
    bars = engine.bar_chords or []
    if not bars:
        return ""
    eng_tick = max(0, tally - _PLAY_LINE_OFFSET)
    return bars[min(eng_tick // 16 + 1, len(bars) - 1)]


# ────────────────────────── Pygame setup ──────────────────────────────
real_pos = [40 + 33.333 * hor_pos[p] * 36/21 + 2 for p in range(37)]

pygame.init()
print(" starting ")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

size = (1280, 720)
screen = pygame.display.set_mode(size, vsync=1)
font  = pygame.font.SysFont('Calibri', 40, True, False)
font1 = pygame.font.SysFont('Calibri', 50, True, False)
pygame.display.set_caption("Jazz")

done = False
clock = pygame.time.Clock()

game = Music(HEIGHT, WIDTH)
counter = 0
tally = 0

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
        # REALTIME STEP — once per 16th-note tick boundary.
        #
        # Display has ~31 ticks of visual lookahead (16 buffer rows + 15
        # ticks of scroll from y=0 to the play line at y=15). The player
        # only plays the model's first prediction at wall-clock tick 31.
        # Until then we MUST NOT commit to the engine — otherwise we'd
        # feed it 30 ticks of "silence" before the player has had a chance
        # to play, which conditions the model into long, dull notes.
        #
        # After tally >= 31, every wall-clock tick we:
        #   1. read the dominant pitch from the SP aggregator over the
        #      last 16th window,
        #   2. commit that token to the engine,
        #   3. replace total_sheet[tally..] from rollout[14..] (upcoming),
        #   4. update_image on-screen figures from rollout[0..13]. THIS
        #      is what makes deviations VISIBLE near the play line.
        # ──────────────────────────────────────────────────────────────
        if not engine.is_done and tally >= 31:
            played_token = get_played_token()
            rollout = engine.commit(played_token)
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
                      f"buf[0:5]={buf_preview}")
            if rollout:
                # Render the FULL rollout in one pass so the prev-pitch chain
                # for HOLD/REST is consistent. `engine.current_pitch` anchors
                # the start (post-deviation that's the deviation note, not
                # whatever stale pitch the previous on-screen row held).
                spawn_prev = (engine.current_pitch
                              if engine.current_pitch is not None
                              else DEFAULT_DISPLAY_PITCH)
                all_rendered, _ = render_tokens(rollout, spawn_prev)

                # --- (a) UPDATE ON-SCREEN FIGURES with rollout[0..13].
                # A figure with source_idx S is at y = tally - 1 - S right
                # now. Mapping into rollout: k = S - tally + 14, valid for
                # 0 <= k < 14. This is what makes the next note at the
                # BOTTOM of the display visibly react to a deviation.
                for fig in game.figure:
                    if fig.source_idx is None:
                        continue
                    k = fig.source_idx - tally + 14
                    if 0 <= k < 14 and k < len(all_rendered):
                        fig.update_image(all_rendered[k])

                # --- (b) REPLACE / EXTEND total_sheet from rollout[14:]
                # for upcoming spawns at this and future boundaries.
                for offs, row in enumerate(all_rendered[14:]):
                    idx = tally + offs
                    if idx < len(total_sheet):
                        total_sheet[idx] = row
                    else:
                        total_sheet.append(row)

        if tally < len(total_sheet):
            game.new_figure(total_sheet[tally], source_idx=tally)
            tally += 1
        if len(game.figure) != 0 and game.figure[0].y > 19:
            game.figure.popleft()
        if len(game.figure) != 0 and game.figure[0].col_idx is None:
            game.figure.popleft()

    if counter % (Testing_variable_for_testing * 16) == 0:
        game.new_beatbar()

    game.go_down()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                game.__init__(HEIGHT, WIDTH)
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                done = True

    screen.blit(background, (0, 0))

    # ── Figures: group consecutive same-column ticks into one tall
    #    rectangle so a sustained note shows as a single block (matches the
    #    variable-length visual from the latest hardware Piano_display). The
    #    per-tick internal model is what allows realtime updates to figures
    #    already on screen — see the update_image loop above. ──
    if game.figure is not None and len(game.figure) > 0:
        figs = list(game.figure)
        i = 0
        while i < len(figs):
            fig = figs[i]
            col = fig.col_idx
            if col is None:
                i += 1
                continue
            # Find the run of consecutive same-column figures from i onward.
            j = i + 1
            while j < len(figs) and figs[j].col_idx == col:
                j += 1
            # Group spans figs[i..j-1]. figs[0] is oldest (highest y), figs[-1]
            # newest (lowest y). Within the group, last (figs[j-1]) is the
            # newest = top of the visual rectangle.
            top_fig = figs[j - 1]
            n_ticks = j - i
            # Skip if entirely off-screen.
            if top_fig.y > 22 or (top_fig.y + n_ticks - 1) < -2:
                i = j
                continue
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

    if game.beat_bars is not None:
        for beat_bar in game.beat_bars:
            if beat_bar > 15:
                continue
            pygame.draw.line(screen, GRAY,
                             [game.x + game.zoom * 0,     game.y + game.zoom * beat_bar - 3],
                             [game.x + game.zoom * WIDTH, game.y + game.zoom * beat_bar - 3])

    # ── Top-of-screen text + blue square on the played key ────────────
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

pygame.quit()
