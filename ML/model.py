import torch
import numpy
import torch.nn as nn
import numpy as np
import json
import pandas as pd

#need to import num_dur from the data processing otherwise cant know
with open('ML\\data\\raw\\metadata.json', 'r') as file:
    dur_data = json.load(file)
with open('ML\\data\\raw\\chord_vocab.json', 'r') as vocab:
    chord_vocab = json.load(vocab)

num_dur = len(dur_data['duration_values'])
print(num_dur)
solos = pd.read_parquet("ML\\data\\raw\\processed_solos.parquet", engine='pyarrow')
seq_length = 64
device = "cpu"
rng = np.random.default_rng(seed=0)



class MusicPredicitonModel(nn.Module):
    def __init__(self,num_chord, dur_embed = 10, hidden_layers = 512, lstm_layers = 1, pitch_classes=12, chord_embed = 32, pitch_class_embed =16):
        super().__init__()
        self.pitch_class_embed = nn.Embedding(pitch_classes+1, pitch_class_embed)
        self.dur_embed = nn.Embedding(num_dur+1, dur_embed)
        self.chord_embed = nn.Embedding(num_chord+1, chord_embed)
        self.input_dim = pitch_class_embed + dur_embed + chord_embed + 1
        self.lstm = nn.LSTM(self.input_dim, hidden_layers, lstm_layers, batch_first = True)
        self.pitch_class = nn.Linear(hidden_layers, pitch_classes)
        self.duration_head = nn.Linear(hidden_layers, num_dur)
       
    def forward(self, prev_pitch, prev_dur, cur_chord, beat_pos, hidden=None):
        x = torch.cat([
            self.pitch_class_embed(prev_pitch),
            self.dur_embed(prev_dur),
            self.chord_embed(cur_chord),
            beat_pos.unsqueeze(-1)
            ], dim=-1)
        out, hidden = self.lstm(x, hidden)
        pitch_out= self.pitch_class(out)
        dur_out = self.duration_head(out)
        return pitch_out, dur_out, hidden
    
def data_prep(seqs, chord_to_id):
    prepped= []
    print("--------------------------------------------")
    seqs2 =  [
    (group['pitch'].tolist(), group['duration_bin'].tolist(), group['chord'].tolist())
    for _, group in seqs.groupby('melid')
        ]
    num_chords = len(chord_to_id)
    #need to know data prep
    for  pitch, dur, chord in seqs2:
        local_pitch_class = [p % 12 for p in pitch]
        COUNT = len(local_pitch_class)
        prev_pitch_class = [12] + local_pitch_class[:-1]
        prev_dur = [num_dur] + dur[:-1]
        cur_chord = [chord_to_id.get(c, num_chords) for c in chord]

        num_beat = []
        time = 0.0
        prev=None
        for i in range(COUNT):
            if chord[i] != prev:
                time = 0.0
                prev = chord[i]
            num_beat.append(min(time / 4.0, 1.0))
            time += dur_data['duration_values'][dur[i]]
        prepped.append({
            "prev_pitch_class": np.array(prev_pitch_class, dtype=np.int64),
            "prev_dur": np.array(prev_dur, dtype=np.int64),
            "cur_chord": np.array(cur_chord, dtype=np.int64),
            "num_beat": np.array(num_beat, dtype=np.float32),
            "target_pitch_class": np.array(local_pitch_class, dtype=np.int64),
            "target_dur": np.array(dur, dtype=np.int64),
        })
    return prepped
    
def train(model, prepped, batch_size, seq_length=64, epochs=3000, lr=0.001):
    pitch_criterion = nn.CrossEntropyLoss()
    duration_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # model.train()
    for epoch in range(epochs):
        total_loss = 0
        bp_ , bd_ , bc_, tp_, td_, bb_ = [], [], [], [] ,[], [] # empty batches
        for _ in range(batch_size):
            x = prepped[rng.integers(len(prepped))]
            j = len(x["target_pitch_class"])
            if j <= seq_length:
                start, length = 0, j
            else:
                start = int(rng.integers(j - seq_length))
                length = seq_length
            end = start+length
            def pad(a, fill=0):
                array = np.full(seq_length, fill, dtype=a.dtype)
                array[:length] = a[start:end]
                return array
            
            bp_.append(pad(x["prev_pitch_class"], fill = 12))
            bd_.append(pad(x["prev_dur"], fill = num_dur))
            bc_.append(pad(x["cur_chord"], fill =0))
            bb_.append(pad(x["num_beat"]))
            tp_.append(pad(x["target_pitch_class"], fill = -100))
            td_.append(pad(x["target_dur"], fill =-100))

            

        prev_pc = torch.from_numpy(np.stack(bp_)).to(device)
        prev_dur = torch.from_numpy(np.stack(bd_)).to(device)
        cur_chord = torch.from_numpy(np.stack(bc_)).to(device)
        beat_pos = torch.from_numpy(np.stack(bb_)).to(device)
        target_pitch_class = torch.from_numpy(np.stack(tp_)).to(device)
        target_dur = torch.from_numpy(np.stack(td_)).to(device)

            
        pitch_class_logits , dur_logits , _ = model(prev_pc, prev_dur, cur_chord, beat_pos)
        loss_pitch = pitch_criterion(pitch_class_logits.reshape(-1, 12), target_pitch_class.reshape(-1))
        loss_dur = duration_criterion(dur_logits.reshape(-1, num_dur), target_dur.reshape(-1))
        loss = loss_pitch + loss_dur
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if epoch >= 2980:
            print(f"epoch {epoch + 1}, loss: {total_loss:.4f}")
    return model

#train
chord_to_id = {c: i for i, c in enumerate(chord_vocab)}
model = MusicPredicitonModel(num_chord=len(chord_vocab))
# # print(solos.loc[:, ~solos.columns.isin(['melid', 't'])])
# prepped = data_prep(solos, chord_to_id)
# trained_model = train(model,prepped=prepped, batch_size=64, epochs= 3000)
# torch.save(trained_model.state_dict(), "model1.pt")




