"""Train the full DSMV architecture on the existing official split."""
import argparse
import math
from pathlib import Path

import torch

from configs.config import load_config, get_device, MODEL_ROOT
from datasets.datasets import create_dataset
from utils.runtime import (
    seed_everything, move_batch, create_models, compute_losses, write_json,
    save_checkpoint, load_checkpoint, check_checkpoint_config, validation_loss,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--bert-dir', default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', default=str(MODEL_ROOT / 'outputs' / 'dsmv'))
    parser.add_argument('--resume', default=None, help='Checkpoint from this paper-aligned copy')
    args = parser.parse_args()
    cfg = load_config(args.cfg, args.data_dir, args.bert_dir)
    seed_everything(cfg.seed)
    device = get_device(args.device)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError('Output directory is not empty; choose a new directory or --resume.')
    if min(cfg.train.snapshot_interval, cfg.train.max_iter, cfg.train.log_interval) <= 0:
        raise ValueError('Training, logging and snapshot intervals must be positive.')
    train_dataset, train_loader = create_dataset(cfg, 'train')
    _, val_loader = create_dataset(cfg, 'val')
    if len(train_loader) == 0:
        raise ValueError('Training split is empty.')
    encoder, decoder = create_models(cfg, train_dataset, device)
    parameters = [p for model in (encoder, decoder) for p in model.parameters() if p.requires_grad]
    if cfg.train.optim.type.lower() != 'adam':
        raise ValueError('The paper configuration uses Adam.')
    optimizer = torch.optim.Adam(parameters, lr=cfg.train.optim.lr,
                                 weight_decay=cfg.train.optim.weight_decay)
    step, best_val = 0, math.inf
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        check_checkpoint_config(checkpoint, cfg, train_dataset)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = cfg.train.optim.lr
        step, best_val = checkpoint['step'], checkpoint['best_val']
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', cfg)
    encoder.train()
    decoder.train()
    while step < cfg.train.max_iter:
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(encoder(batch), batch['input_tokens'])
            total, nll, dice = compute_losses(
                logits, batch['targets'], batch['mask'], cfg.train.loss_alpha)
            total.backward()
            optimizer.step()
            step += 1
            if step % cfg.train.log_interval == 0:
                print(f'step={step} total={total.item():.6f} '
                      f'decoder={nll.item():.6f} dice={dice.item():.6f}', flush=True)
            if step % cfg.train.snapshot_interval == 0 or step == cfg.train.max_iter:
                metrics = validation_loss(encoder, decoder, val_loader, cfg, device)
                improved = metrics['loss'] < best_val
                if improved:
                    best_val = metrics['loss']
                save_checkpoint(output / 'last.pt', cfg, encoder, decoder, optimizer,
                                step, best_val, train_dataset)
                if improved:
                    save_checkpoint(output / 'best.pt', cfg, encoder, decoder, optimizer,
                                    step, best_val, train_dataset)
                write_json(output / f'validation_{step}.json', {'step': step, **metrics})
                print(f'validation step={step}: {metrics}', flush=True)
                encoder.train()
                decoder.train()
            if step >= cfg.train.max_iter:
                break


if __name__ == '__main__':
    main()
