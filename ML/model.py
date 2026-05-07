import torch
import numpy
import torch.nn as nn
import numpy as np
import json

#need to import num_dur from the data processing otherwise cant know
with open('ML\\data\\raw\\metadata.json', 'r') as file:
    dur_data = json.load(file)

num_dur = len(dur_data['duration_values'])

class MusicPredicitonModel(nn.Module):
    def __init__(self,num_chord, dur_embed = num_dur, hidden_layers = 512, lstm_layers = 1, pitch_classes=12, chord_embed = 32):
       self.pitch_class_embed = nn.Embedding(pitch_classes+1, self.pitch_class_embed)
       self.dur_embed = nn.Embedding(num_dur+1, dur_embed)
       self.chord_embed = nn.Embedding(num_chord+1, chord_embed)
       self.input_dim = pitch_classes + dur_embed + chord_embed + 1
       self.lstm = nn.lstm(self.input_dim, hidden_layers, lstm_layers, batch_first = True)
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
    num_chords = len(chord_to_id)
    #need to know data prep
    
def train(model, batch_size, epochs, lr=0.001):
    pitch_criterion = nn.CrossEntropyLoss()
    duration_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for _ in range(batch_size):
            bp_ , bd_ , bc_, tp_, td_, bb_ = [], [], [], [] ,[], [] # empty batches
            #need some more data prep
            optimizer.zero_grad()
            pitch_class_logits , dur_logits , _ = model(prev_pc, prev_dur, cur_chord, beat_pos)
            loss_pitch = pitch_criterion(pitch_class_logits, target_pitch_class)
            loss_dur = duration_criterion(dur_logits, target_dur)
            loss = loss_pitch + loss_dur



            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"epoch {epoch + 1}, loss: {total_loss:.4f}")
        return model

#train
model = MusicPredicitonModel(num_chord=32)
trained_model = train(model, batch_size=64, epochs= 3000)




