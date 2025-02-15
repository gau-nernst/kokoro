# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

    def forward(self, x, input_lengths, mask):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T)
            x.masked_fill_(mask, 0.0)

        for c in self.cnn:
            x = c(x)
            if mask is not None:
                x.masked_fill_(mask, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]
        if x.shape[0] > 1:
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)
        x = x.transpose(-1, -2)  # (B, chn, T)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid,nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        self.lstm.flatten_parameters()

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        if x.shape[0] > 1:
            input_lengths = text_lengths.cpu().numpy()
            x = nn.utils.rnn.pack_padded_sequence(d, input_lengths, batch_first=True, enforce_sorted=False)
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)
        m = m.to(text_lengths.device).unsqueeze(1)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            lstm = nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
            lstm.flatten_parameters()
            self.lstms.append(lstm)
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, masks):
        # x: (B, D, T)
        # style: (1, 128)
        x = x.transpose(1, 2)  # (B, T, D)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        if masks is not None:
            masks = masks.unsqueeze(-1)
            x.masked_fill_(masks, 0.0)
        input_lengths_cpu = text_lengths.cpu()

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x, style)
                x = torch.cat([x, s], dim=-1)
                if masks is not None:
                    x.masked_fill_(masks, 0.0)
            else:
                if x.shape[0] > 1:
                    x = nn.utils.rnn.pack_padded_sequence(x, input_lengths_cpu, batch_first=True, enforce_sorted=False)
                    x, _ = block(x)
                    x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                else:
                    x, _ = block(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
