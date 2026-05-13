import torch
import numpy as np
import pandas as pd
import  model as mod


LSTMmodel = mod.model
LSTMmodel.load_state_dict(torch.load('model1.pt', weights_only=True))



def midi_gen(pitch_class, prev_midi, default, pitch_high, pitch_low, max_step, rng, prob):
    if prev_midi is not None:
        base = prev_midi
    else:
        base = default
    nearest_pitch = base + ((pitch_class- base)%12)
    if nearest_pitch - base >6:
        nearest_pitch -= 12
    
    possibilieties = [nearest_pitch -12, nearest_pitch, nearest_pitch + 12]
    weigths = np.array([prob /2, 1.0 - prob, prob/2])
    for i, p in enumerate(possibilieties):
        if p < pitch_low or p > pitch_high:
            weigths[i] = 0.0
        elif prev_midi is not None and abs(p - prev_midi) > max_step:
            weigths[i] = 0.0
        else:
            weigths[i] *= 1.0 /(1.0+0.05 * abs(p-default))
    if weigths.sum() == 0:
        return int(min(max(nearest_pitch, pitch_low), pitch_high))
    weigths/=weigths.sum()
    return int(possibilieties[int(rng.choice(3, p=weigths))])

@torch.no_grad()
def generate_music(model, chord_to_id, chord_progression, key, max_notes = 600, temperature = 1.0):
    default = 67
    max_step = 12
    prob = 0.05
    pitch_low = 55
    pitch_high = 84
    generated = []
    num_chords = len(chord_to_id)

    chords_in_C = []

    notes= []
    prev_pitch_class = 12
    prev_dur = mod.num_dur
    prev_midi = None
    hidden = None

    for chord_id_pg, (chord, chord_beat) in enumerate(chord_progression):
        cur_chord_id = chord_to_id.get(chord, num_chords)
        elapsed_time = 0.0
        while elapsed_time < chord_beat and len(notes) < max_notes:
            beat_pos = float(min(elapsed_time/4.0, 1.0))
            x_pitch_class = torch.tensor([[prev_pitch_class]])
            x_dur = torch.tensor([[prev_dur]])
            x_chord = torch.tensor([[cur_chord_id]])
            x_bp = torch.tensor([[beat_pos]])

            pitch_class_logits, dur_logits, hidden = model(x_pitch_class, x_dur, x_chord, x_bp, hidden)

            pitch_class_prob = torch.softmax(pitch_class_logits[0,0]/max(temperature, 1e-6), dim=-1).cpu().numpy()
            pitch_class = int(mod.rng.choice(12, p = pitch_class_prob/pitch_class_prob.sum()))

            dur_prob = torch.softmax(dur_logits[0,0]/max(temperature, 1e-6),dim= -1).cpu().numpy()
            dur_prob = dur_prob/dur_prob.sum()
            dur = int(mod.rng.choice(mod.num_dur))

            beats = mod.dur_data['duration_values'][dur]
            remain = chord_beat - elapsed_time
            if beats > remain + 1e-6:
                beats = remain
            
            midi = midi_gen(pitch_class, prev_midi, default, pitch_high, pitch_low, max_step, mod.rng, prob)

            notes.append((midi, beats, chord_id_pg))
            elapsed_time += beats
            prev_pitch_class = pitch_class
            prev_dur = dur
            prev_midi = midi
    output = []
    for midi , beats, chord_id_pg in notes:
        output.append((int(midi), float(beats), chord_progression[chord_id_pg][0]))
    return output
            
BPM = 80
iiVI_cycle = [("Cm7", 4), ("F7", 4), ("Bbmaj7", 4), ("Bbmaj7", 4)]
iiVI_long = iiVI_cycle * 5
notes = generate_music(LSTMmodel, mod.chord_to_id, iiVI_long, temperature=0.95, key= "Bb")
print(len(notes))
    

    