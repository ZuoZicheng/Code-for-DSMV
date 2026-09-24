"""Shared train/evaluation plumbing for the paper-aligned copy."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.modules import ChangeDetector
from models.dynamic_speaker_change_pos import DynamicSpeaker
from models.transformer import DiceLoss


CHECKPOINT_FORMAT = 'dsmv-paper-aligned-v1'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def create_models(cfg, dataset, device):
    encoder = ChangeDetector(cfg, dataset.word_to_idx).to(device)
    decoder = DynamicSpeaker(cfg, dataset.vocab_size, dataset.bos_id).to(device)
    return encoder, decoder


def compute_losses(logits, targets, mask, alpha):
    token_loss = F.cross_entropy(logits.transpose(1, 2), targets.long(), reduction='none')
    decoder_loss = (token_loss * mask).sum() / mask.sum().clamp_min(1)
    dice_loss = DiceLoss()(logits, targets, mask)
    return alpha * decoder_loss + dice_loss, decoder_loss, dice_loss


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def save_checkpoint(path, cfg, encoder, decoder, optimizer, step, best_val, dataset):
    torch.save({
        'format': CHECKPOINT_FORMAT, 'config': json.loads(json.dumps(cfg)),
        'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(), 'step': step, 'best_val': best_val,
        'word_to_idx': dataset.word_to_idx,
    }, path)


def load_checkpoint(path, device):
    # This format contains plain mappings/tensors, not pickled model objects.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Old EKAID/DSMV checkpoints are not compatible with this architecture.')
    return checkpoint


def check_checkpoint_config(checkpoint, cfg, dataset):
    if checkpoint['config']['model'] != cfg.model:
        raise ValueError('Checkpoint model configuration differs from the selected YAML.')
    if checkpoint['word_to_idx'] != dataset.word_to_idx:
        raise ValueError('Checkpoint and dataset vocabularies differ.')


@torch.no_grad()
def validation_loss(encoder, decoder, loader, cfg, device):
    encoder.eval()
    decoder.eval()
    nll_sum = 0.0
    token_count = 0.0
    dice_sum = 0.0
    examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = decoder(encoder(batch), batch['input_tokens'])
        _, nll, dice = compute_losses(logits, batch['targets'], batch['mask'], cfg.train.loss_alpha)
        count = batch['mask'].sum().item()
        size = batch['targets'].size(0)
        nll_sum += nll.item() * count
        token_count += count
        dice_sum += dice.item() * size
        examples += size
    if examples == 0 or token_count == 0:
        raise ValueError('Validation split has no eligible examples.')
    nll = nll_sum / token_count
    dice = dice_sum / examples
    return {'loss': cfg.train.loss_alpha * nll + dice, 'decoder_loss': nll, 'dice_loss': dice}


def decode_tokens(sequence, idx_to_word, bos_id):
    words = []
    for token in sequence:
        token = int(token)
        if token == 0:
            break
        if token != bos_id:
            words.append(idx_to_word[token])
    return ' '.join(words)
