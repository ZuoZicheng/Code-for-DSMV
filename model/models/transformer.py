"""DSMV: question-independent image comparison, then question-guided fusion.

Batch-first throughout. Layer wiring choices are documented in PAPER_ALIGNMENT.md.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer


class DiceLoss(nn.Module):
    """Masked per-example Dice, averaged over the batch (manuscript Eq. 15)."""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets, mask):
        probabilities = logits.softmax(dim=-1)
        one_hot = F.one_hot(targets.long(), num_classes=logits.size(-1))
        one_hot = one_hot.to(probabilities.dtype)
        mask = mask.unsqueeze(-1).to(probabilities.dtype)
        intersection = (probabilities * one_hot * mask).sum(dim=(1, 2))
        denominator = ((probabilities + one_hot) * mask).sum(dim=(1, 2))
        return (1 - (2 * intersection + self.epsilon) /
                (denominator + self.epsilon)).mean()


class PositionWiseFFN(nn.Module):
    def __init__(self, width, hidden_width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, hidden_width), nn.ReLU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, x):
        return self.layers(x)


class AddNorm(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, residual, update):
        return self.norm(residual + self.dropout(update))


class ImageInteractionLayer(nn.Module):
    """MVMHA: both directional updates read the same pre-layer image pair."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.before_query = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.after_query = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.before_norm = AddNorm(width, dropout)
        self.after_norm = AddNorm(width, dropout)

    def forward(self, before, after):
        before_update = self.before_query(before, after, after, need_weights=False)[0]
        after_update = self.after_query(after, before, before, need_weights=False)[0]
        return (self.before_norm(before, before_update),
                self.after_norm(after, after_update))


class GlobalFeatureLearning(nn.Module):
    def __init__(self, width, hidden_width, dropout, kernel_size, stride, padding):
        super().__init__()
        self.ffn = PositionWiseFFN(2 * width, hidden_width)
        self.norm = AddNorm(2 * width, dropout)
        self.conv = nn.Conv1d(2 * width, 2 * width, kernel_size,
                              stride=stride, padding=padding)
        self.projection = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Dropout(dropout))
        # Registered once: included in state_dict and optimizer parameters.
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, before, after):
        difference = torch.cat([before, after], dim=-1)  # B x L x 2d
        difference = self.norm(difference, self.ffn(difference))
        local = F.relu(self.conv(difference.transpose(1, 2))).transpose(1, 2)
        local = self.projection(local)  # B x Lv x d
        return torch.cat([self.cls_token.expand(local.size(0), -1, -1), local], dim=1)


class MultimodalFusionLayer(nn.Module):
    def __init__(self, width, heads, hidden_width, dropout):
        super().__init__()
        self.visual_query = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.question_query = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.visual_attn_norm = AddNorm(width, dropout)
        self.question_attn_norm = AddNorm(width, dropout)
        self.visual_ffn = PositionWiseFFN(width, hidden_width)
        self.question_ffn = PositionWiseFFN(width, hidden_width)
        self.visual_ffn_norm = AddNorm(width, dropout)
        self.question_ffn_norm = AddNorm(width, dropout)

    def forward(self, visual, question, question_padding_mask):
        # The padding mask belongs to the key/value sequence, not the query.
        visual_update = self.visual_query(
            visual, question, question, key_padding_mask=question_padding_mask,
            need_weights=False)[0]
        question_update = self.question_query(
            question, visual, visual, need_weights=False)[0]
        visual = self.visual_attn_norm(visual, visual_update)
        question = self.question_attn_norm(question, question_update)
        visual = self.visual_ffn_norm(visual, self.visual_ffn(visual))
        question = self.question_ffn_norm(question, self.question_ffn(question))
        question = question.masked_fill(question_padding_mask.unsqueeze(-1), 0)
        return visual, question


class TransformerEncoder(nn.Module):
    def __init__(self, cfg, word_to_idx):
        super().__init__()
        options = cfg.model.dsmv
        width = options.d_model
        self.idx_to_word = {int(index): word for word, index in word_to_idx.items()}
        self.question_length = options.question_length
        self.tokenizer = BertTokenizer.from_pretrained(
            cfg.paths.bert_dir, local_files_only=True)
        self.bert = BertModel.from_pretrained(cfg.paths.bert_dir, local_files_only=True)
        self.bert.requires_grad_(False)
        self.bert.eval()
        if self.bert.config.hidden_size != width:
            raise ValueError('Use BERT-large with hidden_size equal to d_model=1024.')
        self.image_layers = nn.ModuleList([
            ImageInteractionLayer(width, options.image_heads, options.dropout)
            for _ in range(options.num_layers)
        ])
        self.global_features = GlobalFeatureLearning(
            width, options.visual_ffn_hidden, options.dropout,
            options.conv_kernel, options.conv_stride, options.conv_padding)
        self.fusion_layers = nn.ModuleList([
            MultimodalFusionLayer(width, options.fusion_heads,
                                  options.fusion_ffn_hidden, options.dropout)
            for _ in range(options.num_layers)
        ])

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        return self

    def encode_question(self, token_ids, device):
        texts = []
        for row in token_ids.detach().cpu().tolist():
            words = []
            for token in row:
                if token == 0:
                    break
                word = self.idx_to_word[int(token)]
                if word not in ('<start>', '<end>', '<pad>'):
                    words.append(word)
            texts.append(' '.join(words))
        encoded = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.question_length, return_tensors='pt').to(device)
        with torch.no_grad():
            question = self.bert(**encoded).last_hidden_state
        return question, encoded['attention_mask'].eq(0)

    def forward(self, before, after, question_ids):
        # Finish ALL image-only layers before introducing question features.
        for layer in self.image_layers:
            before, after = layer(before, after)
        visual = self.global_features(before, after)
        question, padding_mask = self.encode_question(question_ids, visual.device)
        for layer in self.fusion_layers:
            visual, question = layer(visual, question, padding_mask)
        # No 2d -> d compression before the decoder: manuscript z is B x 2d.
        return torch.cat([visual[:, 0, :], question[:, 0, :]], dim=-1)
