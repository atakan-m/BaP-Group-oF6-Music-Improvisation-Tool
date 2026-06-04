"""Render a few stylistically-varied solos into ML/GeneratedSolos/.

Run from the project root:
    python ML/GeneratedSolos/generate_batch.py

Each entry below becomes one .mid file. Tweak the dict to taste, or just
use the CLI directly (see the comment block at the bottom of this file).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GENERATE = os.path.join(PROJECT_ROOT, "ML", "generate.py")

# Each entry: filename -> list of CLI args
SOLOS = {
    # --- ii-V-I in C major, three temperaments ----------------------
    "iiVI_C_ballad.mid": [
        "--chords", "Dm7,G7,Cj7,Cj7",
        "--bars", "16", "--temperature", "0.6",
        "--hold_penalty", "-0.4",
        "--rest_penalty", "1.0",
        "--seed", "1",
    ],
    "iiVI_C_neutral.mid": [
        "--chords", "Dm7,G7,Cj7,Cj7",
        "--bars", "16", "--temperature", "0.8",
        "--hold_penalty", "0.0",
        "--rest_penalty", "1.0",
        "--seed", "2",
    ],
    "iiVI_C_bebop.mid": [
        "--chords", "Dm7,G7,Cj7,Cj7",
        "--bars", "16", "--temperature", "0.9",
        "--hold_penalty", "1.0",
        "--rest_penalty", "1.5",
        "--seed", "3",
    ],

    # --- minor ii-V-i ----------------------------------------------
    "minor_iiVi_ballad.mid": [
        "--chords", "Dm7b5,G7,Cm7,Cm7",
        "--bars", "16", "--temperature", "0.5",
        "--hold_penalty", "-0.5",
        "--rest_penalty", "1.5",
        "--seed", "4",
    ],

    # --- 12-bar blues in C -----------------------------------------
    "blues_C.mid": [
        "--chords", "C7,F7,C7,C7,F7,F7,C7,C7,G7,F7,C7,G7",
        "--bars", "24", "--temperature", "0.8",
        "--hold_penalty", "0.5",
        "--rest_penalty", "1.0",
        "--seed", "5",
    ],

    # --- rhythm changes A-section ---------------------------------
    "rhythm_changes_A.mid": [
        "--chords", "C,A7,Dm7,G7,Em7,A7,Dm7,G7",
        "--bars", "32", "--temperature", "0.85",
        "--hold_penalty", "0.5",
        "--rest_penalty", "1.5",
        "--seed", "6",
    ],
}


def main():
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output folder: {HERE}\n")
    for name, args in SOLOS.items():
        out_path = os.path.join(HERE, name)
        cmd = [sys.executable, GENERATE, "--out", out_path] + args
        # The args list already has dashes; show a readable summary
        readable = " ".join(args[i] + "=" + args[i+1] for i in range(0, len(args), 2))
        print(f"▶  {name}")
        print(f"   {readable}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   FAILED:\n{result.stderr}")
        else:
            # generate.py prints e.g. "wrote ... (N ticks)"
            for line in result.stdout.strip().splitlines():
                print(f"   {line}")
        print()


if __name__ == "__main__":
    main()


# ======================================================================
# DIRECT CLI USAGE (one-off solos without editing this script)
# ======================================================================
#
# From the project root, the most basic invocation is:
#
#   python ML/generate.py --chords "Dm7,G7,Cj7,Cj7" --bars 16 \
#                         --out ML/GeneratedSolos/my_solo.mid
#
# All the tunable hyperparameters:
#
#   --chords         Comma-separated chord progression, expressed *as if in C*.
#                    Examples:
#                      ii-V-I major   : "Dm7,G7,Cj7,Cj7"
#                      ii-V-i minor   : "Dm7b5,G7,Cm7,Cm7"
#                      12-bar blues   : "C7,F7,C7,C7,F7,F7,C7,C7,G7,F7,C7,G7"
#                    Each chord lasts ONE BAR; the progression loops to fill
#                    --bars.
#
#   --bars           Total song length in bars (default 16 → ~48 s @ 80 BPM).
#
#   --temperature    Sampling sharpness.
#                      0.5  smooth, predictable
#                      0.8  default — balanced
#                      1.2  noisy, surprising
#
#   --hold_penalty   Negative → longer notes (ballad). Positive → shorter
#                    (bebop). 0 = training distribution. Range typically
#                    [-1.0, +1.5].
#
#   --rest_penalty   Higher → fewer rests, denser line. Typically [0.5, 2.0].
#
#   --seed           Same seed + same args = same notes (reproducible).
#                    Change for a different rendition of the same plan.
#
#   --tempo_bpm      MIDI playback tempo (default 80).
#
#   --out            Output path. Conventionally
#                    ML/GeneratedSolos/<descriptive_name>.mid
#
# Tiny examples:
#
#   # fast bebop run over rhythm-changes A
#   python ML/generate.py \
#     --chords "C,A7,Dm7,G7,Em7,A7,Dm7,G7" \
#     --bars 32 --temperature 0.9 --hold_penalty 1.0 \
#     --out ML/GeneratedSolos/my_bebop.mid
#
#   # slow ballad over a single chord (modal sketch)
#   python ML/generate.py \
#     --chords "Cm7" --bars 16 \
#     --temperature 0.5 --hold_penalty -0.7 \
#     --out ML/GeneratedSolos/cm7_ballad.mid
