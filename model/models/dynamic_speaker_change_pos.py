"""LSTM answer decoder conditioned only on DSMV z and preceding tokens.

Retains the original decoder's recurrent family, word width and hidden width.
The exact z-to-LSTM wiring is an implementation choice, documented separately.
"""
import torch
from torch import nn

from .CaptionModel import CaptionModel


class DynamicSpeaker(CaptionModel):
    def __init__(self, cfg, vocab_size, bos_id, eos_id=0):
        super().__init__()
        options = cfg.model.speaker
        if options.conditioning_dim != 2 * cfg.model.dsmv.d_model:
            raise ValueError('The decoder must receive z with width 2d.')
        self.seq_length = options.seq_length
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, options.word_embed_size, padding_idx=0),
            nn.ReLU(), nn.Dropout(options.drop_prob_lm))
        self.lstm = nn.LSTM(
            options.word_embed_size + options.conditioning_dim,
            options.rnn_size, num_layers=options.rnn_num_layers,
            batch_first=True,
            dropout=options.drop_prob_lm if options.rnn_num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(options.drop_prob_lm)
        self.output = nn.Linear(options.rnn_size, vocab_size)

    def _forward(self, z, input_tokens):
        words = self.embedding(input_tokens.long())
        condition = z.unsqueeze(1).expand(-1, words.size(1), -1)
        hidden, _ = self.lstm(torch.cat([words, condition], dim=-1))
        return self.output(self.dropout(hidden))  # unnormalized B x T x C logits

    @torch.no_grad()
    def _sample(self, z):
        """Greedy decoding. Ground-truth answer tokens are never an input."""
        batch_size = z.size(0)
        tokens = torch.full((batch_size,), self.bos_id, device=z.device, dtype=torch.long)
        result = torch.full((batch_size, self.seq_length), self.eos_id,
                            device=z.device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z.device)
        state = None
        for step in range(self.seq_length):
            words = self.embedding(tokens).unsqueeze(1)
            hidden, state = self.lstm(torch.cat([words, z.unsqueeze(1)], dim=-1), state)
            logits = self.output(hidden[:, 0, :])
            logits[:, self.bos_id] = -torch.inf
            tokens = logits.argmax(dim=-1)
            tokens = tokens.masked_fill(finished, self.eos_id)
            result[:, step] = tokens
            finished = finished | tokens.eq(self.eos_id)
            if bool(finished.all()):
                break
        return result
