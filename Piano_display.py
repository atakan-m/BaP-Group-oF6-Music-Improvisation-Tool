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
import bapv1                      # kept for the legacy `inputty` reference below
import thepipeforrainier          # new SP pitch detector
import Buttons_v2
import sounddevice as sd

SAMPLE_RATE = thepipeforrainier.SAMPLE_RATE     # 44100
HOP_SIZE    = thepipeforrainier.HOP_SIZE        # 512 = ~11.6 ms/callback
WIDTH = 36
HEIGHT = 20
fps = 60 #bpm
bpm = 60
Testing_variable_for_testing = (fps*fps//bpm//4)
speed = 1/Testing_variable_for_testing

# New detector: exposes `detector.last_midi` (int or None) on every audio block
detector = thepipeforrainier.PitchDetector()

hor_pos = [0,0.5,1,1.5,2,3,3.5,4,4.5,5,5.5,6,7,7.5,8,8.5,9,10,10.5,11,11.5,12,12.5,13,14,14.5,15,15.5,16,17,17.5,18,18.5,19,19.5,20]
key_width = [1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1,1,0.5,1,0.5,1,1,0.5,1,0.5,1,0.5,1]
buffer_sheet = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         ]

sheet = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ]

col_start = 48
#notes = generate.generate_music(generate.LSTMmodel, generate.chord_to_id, generate.iiVI_long, temperature=0.95)
def note_to_col(note):
    col = note - col_start
    return col

def notes_in_row(notes, num_cols= 36):
    row = [0]*num_cols
    col = note_to_col(notes[0])
    # NOTE: col_start=48, so col=0 means MIDI 48 (C3). Using row[col]
    # (not row[col-1]) aligns the dot with the C-labeled visual column
    # at position 0 — previously this was off by one semitone.
    if 0 <= col < num_cols:
        row[col] = 1
    return row

def make_sheet(notes, num_cols=36 ):
    sheet = [notes_in_row(note, num_cols)for note in notes]
    return sheet


inputty = set()
if len(bapv1.PureArrayDetector().melody) > 0:
    inputty = {bapv1.PureArrayDetector().melody[0]}

figures = [
        1,2,3,4,5,6,7,8,9,10,11,12,
         13,14,15,16,17,18,19,20,21,22,23,24,
         25,26,27,28,29,30,31,32,33,34,35,36
        ]
Note_list = [
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    ]
x = 0 
def check_note(game, inputty, x):
    setty = set()
    if len(game.figure) != 0:
        for i in game.figure[0].image:
            if i:
                setty.add(i)
            if game.figure[0].y > 15 and inputty.issubset(setty):
                game.figure.popleft()
                return True
            #if game.figure[0].y > 15:

    return None

class Figure:

    def __init__(self, Notes, source_idx=None):
        self.x = 0 #note
        self.y = 0
        self.tune = Notes #Name of note
        # The total_sheet index this figure was spawned from. Used so the
        # realtime engine can find and update on-screen figures when the
        # model re-plans after a deviation — without this, only future
        # spawns would visibly change.
        self.source_idx = source_idx

        self._image = [figures[i] if Notes[i] != 0 else '' for i in range(len(Notes))]

        self._note_names = [Note_list[i] if Notes[i] != 0 else '' for i in range(len(Notes))]

    @property
    def image(self):
        return self._image

    @property
    def noteName(self):
        return self._note_names

    def update_image(self, Notes):
        """Re-render this figure's image from a new row. Called after a
        rollout regenerate so the figures already scrolling toward the
        play line visibly reflect the new plan instead of the old one."""
        self.tune = Notes
        self._image = [figures[i] if Notes[i] != 0 else '' for i in range(len(Notes))]
        self._note_names = [Note_list[i] if Notes[i] != 0 else '' for i in range(len(Notes))]


class Music:
    def __init__(self, height, width):
        self.score = 0
        self.x = 40
        self.y = 0
        self.zoom = 33.333333
        self.figure = deque()
        self.fuck = False
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
            
            
#get the chord progression and spin up the realtime re-planning engine
chord_prog_2 = []
for chord in Buttons_v2.main():
    chord_prog_2.append((chord, 1))   # 1 bar per chord (engine wants bars)

engine = generate.make_realtime_engine(
    chord_prog_2,
    total_bars=64,
    # --- BALLAD-STYLE TUNING ---
    # Lower temperature => smoother, less jittery sampling. 0.6 still has
    # enough variety to improvise; bump to 0.8 for more surprise notes,
    # drop to 0.4 for very "by-the-book" ballad lines.
    temperature=0.45,
    rollout_ticks=30,            # ~5.6s of lookahead — covers display window
    # --- DEVIATION = QUARTER NOTE ---
    # The deviation lock simulates the LSTM through `NOTE_X + (N-1) HOLDs`
    # before resuming, which pre-loads a "sustain" attractor into the
    # model's hidden state. With deviation_lock_ticks=8 the half-note
    # context made the model want to keep holding for many more ticks
    # after the lock expired — most of the "super long notes" came from
    # that. Quarter (4) gives a less aggressive attractor seed while
    # still being visible as a sustained note.
    deviation_lock_ticks=4,
    # ──── SINGLE BALLAD SLIDER ────────────────────────────────────────
    # `hold_penalty` is the one number to tune for overall feel.
    # Negative values bias the model toward HOLD tokens (longer notes,
    # ballad feel); positive values bias toward NOTE attacks (shorter
    # notes, bebop feel). At 0 the model uses its training distribution
    # (~29% HOLDs, average note ≈ 8th-note).
    #
    # Suggested values:
    #   +1.0   fast bebop (lots of attacks, ~16th-note feel)
    #    0.0   neutral (~8th-note feel)
    #   -0.3   mild ballad (current — quarter-note avg, occasional sustains)
    #   -0.7   strong ballad (notes routinely a half-note long)
    #   -1.0+  drone (whole-notes, can hang on one pitch a long time —
    #          especially after a deviation lock primes 8 HOLDs into the
    #          context; the model then keeps holding indefinitely)
    # ─────────────────────────────────────────────────────────────────
    hold_penalty=0.7,
    # Keep REST suppressed so the model doesn't fall into silence — ballads
    # have rests but not minute-long silences. Bump to 2.5 if you see too
    # many rests; drop to 0.5 to allow more rests/breath between phrases.
    rest_penalty=1.5,
    # ──── HARD CAP ON CONSECUTIVE HOLDS ────────────────────────────────
    # The soft `hold_penalty` shifts probabilities, but at low temperature
    # a strongly held attractor (e.g. mid-sustain context where the model's
    # raw HOLD logit is +2 above NOTE) can still win even with large
    # penalties. This is a hard upper bound: after this many HOLDs in a
    # row, the next sample is forbidden from being HOLD — guaranteed
    # maximum note length = (max_consec_holds + 1) ticks. Set to None to
    # disable the cap.
    #   4   → max 5-tick notes ≈ quarter note (current — quite fast)
    #   8   → max 9-tick notes ≈ half note (moderate)
    #   16  → max 17-tick notes ≈ whole note (ballad-friendly)
    max_consec_holds=4,
)

# Set to True to see commit + match/deviation logs in the terminal.
DEBUG_REALTIME = True

# --- Display rendering: token (REST/HOLD/NOTE_x) -> 36-col binary row ---
# Display can't draw real rests (notes_in_row always sets one bit), so REST
# and HOLD tokens reuse the previously sounding pitch. We render non-statefully
# so we can re-render the *future* of total_sheet on every commit without
# corrupting an internal "previous pitch" global.
PITCH_LOW_DISPLAY  = 48              # MIDI 48 = C3 (Note_list[0] = "C")
PITCH_HIGH_DISPLAY = 83              # MIDI 83 = B5 (Note_list[35] = "B")
DEFAULT_DISPLAY_PITCH = 60


def render_tokens(tokens, start_prev_midi):
    """Render a sequence of tokens into display rows.

    Returns (rows, final_prev_midi). Pass `final_prev_midi` back in as
    `start_prev_midi` next time to continue the chain.
    """
    rows = []
    prev = start_prev_midi
    for tok in tokens:
        tok = int(tok)
        if tok >= 2:                              # NOTE_<midi>
            midi = 40 + (tok - 2)
            if midi < PITCH_LOW_DISPLAY:  midi = PITCH_LOW_DISPLAY
            if midi > PITCH_HIGH_DISPLAY: midi = PITCH_HIGH_DISPLAY
            prev = midi
        rows.append(notes_in_row((prev,), 36))
    return rows, prev


def row_to_midi(row):
    """Extract the MIDI pitch encoded in a 36-col display row. (Inverse of
    `notes_in_row((midi,), 36)`.)"""
    for i, v in enumerate(row):
        if v:
            return 48 + i                         # row[col] with col_start=48
    return DEFAULT_DISPLAY_PITCH


# Pre-fill the display with the engine's initial 10-second rollout so the
# player can read upcoming notes before the first commit.
initial_rollout = engine.rollout()
sheet, _prev_chain_midi = render_tokens(initial_rollout, DEFAULT_DISPLAY_PITCH)
total_sheet = buffer_sheet + sheet

# Per-16th-note aggregator for SP detections.
# The SP audio callback fires every ~46 ms; we accumulate every detection
# inside the current 16th-note window and pick the dominant pitch + decide
# REST/HOLD/NOTE_x once at each tick boundary.
mic_aggregator = generate.MicTickAggregator()

# Wrap the detector's callback so each audio block also feeds the aggregator.
# MUST be done before the InputStream below is started.
_original_detector_callback = detector.callback
def _detector_callback_with_aggregation(indata, frames, time_info, status):
    _original_detector_callback(indata, frames, time_info, status)
    # The new PitchDetector exposes `last_midi` as int (or None for silence).
    mic_aggregator.observe(detector.last_midi)
detector.callback = _detector_callback_with_aggregation

def get_played_token():
    """Return the engine token for the just-finished 16th-note window."""
    return mic_aggregator.consume_token()

real_pos = [40 + 33.333 * hor_pos[p] * 36/21 + 2 for p in range(36)]

# Initialize the game engine
pygame.init()
print(" starting ")

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

size = (1280, 720) # width, height
flags = pygame.FULLSCREEN
screen = pygame.display.set_mode(size, flags, vsync=1)
font = pygame.font.SysFont('Calibri', 40, True, False)
font1 = pygame.font.SysFont('Calibri', 50, True, False)
pygame.display.set_caption("Jazz")

# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()

game = Music(HEIGHT, WIDTH) #height, width
counter = 0
tally = 0
pressing_down = False

background = pygame.Surface(size)
background.fill(WHITE)

for i in range(game.height):
    for j in range(21 + 1):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 36/21, game.y + game.zoom * i, 1, game.zoom], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 1.4, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 3.25, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 6.55, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 8.33, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    for j in range(3):
        pygame.draw.rect(background, GRAY, [game.x + game.zoom * j * 12 + game.zoom * 10.11, game.y + game.zoom * 18 + game.zoom * -3, game.zoom * 0.5, game.zoom* 3], 1)
    pygame.draw.line(background, GRAY, [game.x + game.zoom * 0, game.y + game.zoom* 0], [game.x + game.zoom* WIDTH, game.y + game.zoom * 0] )
    pygame.draw.line(background, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * 20], [game.x + game.zoom* WIDTH, game.y + game.zoom * 20] )
    pygame.draw.line(background, RED, [game.x + game.zoom * 0, game.y + game.zoom * 15], [game.x + game.zoom* WIDTH, game.y + game.zoom * 15] )

sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                blocksize=HOP_SIZE, dtype="float32",
                callback=detector.callback).start()


while not done:
    counter += 1
    if counter > 100000:
        counter = 0

    if counter % (Testing_variable_for_testing) == 0 or pressing_down:
        # ------------------------------------------------------------------
        # REALTIME STEP — once per 16th-note tick boundary.
        #
        # Display has ~31 ticks of visual lookahead (16 buffer rows + 15
        # ticks of read time from y=0 to the play line at y=15). So the
        # player only plays the model's first prediction at wall-clock
        # tick 31. Until then we MUST NOT commit to the engine — otherwise
        # we'd be feeding it 30 ticks of "silence" before the player has
        # had a chance to play anything, which conditions the model into
        # generating long, dull notes.
        #
        # After tally >= 31, every wall-clock tick we:
        #   1. read the dominant pitch from the SP aggregator over the
        #      last 16th window,
        #   2. commit that token to the engine (advances persistent state),
        #   3. extend total_sheet by one new far-future row, AND replace
        #      total_sheet[tally..tally+38] with renders of rollout[14..52].
        #      The replace is what makes deviations VISIBLE: the new
        #      predictions enter the player's view within ~3 s of the
        #      deviation instead of ~10 s later.
        # ------------------------------------------------------------------
        if not engine.is_done and tally >= 31:
            played_token = get_played_token()
            rollout = engine.commit(played_token)
            if DEBUG_REALTIME:
                # Engine reports its own action — no more buggy diffing.
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
                # A figure spawned at source_idx S has y = tally - S at this
                # boundary (before this tick's spawn). For it to map to a
                # rollout entry, S >= tally - 14 (so the prediction is for an
                # engine tick still in the rollout's window) and S < tally
                # (so it's a previously-spawned figure, not the one about
                # to be spawned). The rollout index is k = S - tally + 14.
                # This is what makes the next note at the BOTTOM of the
                # display (about to be played) visibly react to a deviation.
                for fig in game.figure:
                    if fig.source_idx is None:
                        continue
                    k = fig.source_idx - tally + 14
                    if 0 <= k < 14 and k < len(all_rendered):
                        fig.update_image(all_rendered[k])

                # --- (b) REPLACE / EXTEND total_sheet with rollout[14..]
                # for the upcoming spawns at this and future tick boundaries.
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
        if len(game.figure) != 0 and not any(game.figure[0].image):
            game.figure.popleft()

        #if not game.fuck and (game.score - 1) >= 0:
            #game.score -= 1
        game.fuck = False
    
    if counter % (Testing_variable_for_testing*Testing_variable_for_testing) == 0:
        game.new_beatbar()
            

    game.go_down()
    if check_note(game, inputty, game.score) and inputty and counter > 16 :
        game.score += 1
    #else:
        #game.score -= 1
            
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                game.__init__(HEIGHT, WIDTH)
            if event.key == pygame.K_ESCAPE:
                done = True
            if event.key == pygame.K_LEFT:
                if len(game.figure) > 0 and game.figure[0].y > 15 and game.figure[0].x == 0:
                    game.score += 1
                    game.figure.popleft()
                elif game.score - 1 > 0:
                    game.score -= 1
                    game.fuck = True

    screen.blit(background, (0,0))
    
    if game.figure is not None:
        for k in range(len(game.figure)):
            figure = game.figure[k]
            if figure.y < -2 or figure.y > 21:
                continue
            for p in range(len(figure.image)):
                if game.figure[k].image[p]:
                    pygame.draw.rect(screen, (0, 100 * key_width[p], 0), #screen and color
                        [real_pos[p] + game.zoom * (0.5/key_width[p] - 0.5), #x value of rectangle
                        game.y + game.zoom * (game.figure[k].y - 1) + 1, #y value of rectangle
                        1.7* game.zoom * key_width[p] - 3, #Width of rectangle
                        max(game.zoom, game.zoom - 3 * game.figure[k].tune[p]) - 3]) #Height of rectangle
                if game.figure[k-1].noteName[p] != game.figure[k].noteName[p]:
                    screen.blit(font.render(game.figure[k].noteName[p], True, RED), [real_pos[p] + (game.zoom *0.5), 
                        game.y + game.zoom * (game.figure[k].y - 1)])
    if game.beat_bars is not None:
        for k in range(len(game.beat_bars)):
            pygame.draw.line(screen, GRAY, [game.x + game.zoom * 0, game.y + game.zoom * game.beat_bars[k] - 3], [game.x + game.zoom* WIDTH, game.y + game.zoom * game.beat_bars[k] - 3] )

    text = font1.render("Score: " + str(game.score), True, BLACK)
    text1 = font1.render("BPM: " + str(bpm), True, BLACK)
    text2 = font1.render("Time: " + str(counter//fps), True, BLACK)

    screen.blit(text, [0, 0])
    screen.blit(text1, [game.zoom * 15 ,0])
    screen.blit(text2, [game.zoom * 30 ,0])
    #text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))
    

    pygame.display.flip()
    clock.tick(fps)


pygame.quit()
 