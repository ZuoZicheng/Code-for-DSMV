"""Generate answers; evaluate Difference and non-Difference questions separately."""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from configs.config import load_config, get_device, MODEL_ROOT
from datasets.datasets import create_dataset
from utils.runtime import (
    seed_everything, move_batch, create_models, load_checkpoint,
    check_checkpoint_config, decode_tokens, write_json,
)


def normalize_exact_match(text):
    # Explicit local protocol, NOT claimed to reproduce PLURAL normalization.
    return ' '.join(str(text).lower().split())


def evaluate_predictions(predictions, output):
    difference = [row for row in predictions if row['question_type'] == 'difference']
    metrics = {'difference_count': len(difference)}
    if difference:
        # Build references from the matching CSV rows; no category mixing.
        from pycocotools.coco import COCO
        from evaluation import my_COCOEvalCap
        references = {
            'info': {}, 'licenses': [], 'categories': [],
            'images': [{'id': row['qa_id']} for row in difference],
            'annotations': [dict(id=i, image_id=row['qa_id'], caption=row['reference'])
                            for i, row in enumerate(difference)],
        }
        results = [dict(image_id=row['qa_id'], caption=row['prediction']) for row in difference]
        reference_path = output / 'difference_references.json'
        result_path = output / 'difference_predictions.json'
        write_json(reference_path, references)
        write_json(result_path, results)
        coco = COCO(str(reference_path))
        evaluation = my_COCOEvalCap(coco, coco.loadRes(str(result_path)))
        evaluation.evaluate()
        metrics['difference_caption_metrics'] = {key: float(value) for key, value in evaluation.eval.items()}
    eligible = {'presence', 'abnormality', 'location', 'level', 'view', 'type'}
    non_difference = [row for row in predictions if row['question_type'] in eligible]
    buckets = {'open': [], 'closed': [], 'overall': []}
    for row in non_difference:
        reference = normalize_exact_match(row['reference'])
        predicted = normalize_exact_match(row['prediction'])
        group = 'closed' if reference in ('yes', 'no') else 'open'
        match = int(predicted == reference)
        buckets[group].append(match)
        buckets['overall'].append(match)
    metrics['non_difference_exact_match'] = {
        'protocol': 'Lowercase and collapse whitespace; retain punctuation. Closed iff reference is yes/no.',
        'benchmark_comparable': False,
        **{name: {'count': len(values), 'accuracy': sum(values) / len(values) if values else None}
           for name, values in buckets.items()},
    }
    metrics['unrecognized_question_type_count'] = sum(
        row['question_type'] not in eligible | {'difference'} for row in predictions)
    return metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--checkpoint', required=True)
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--bert-dir', default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output-dir', default=str(MODEL_ROOT / 'outputs' / 'evaluation'))
    parser.add_argument('--predictions-only', action='store_true', help='Skip metric computation')
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Evaluation output is not empty; choose a new output directory.')
    cfg = load_config(args.cfg, args.data_dir, args.bert_dir)
    seed_everything(cfg.seed)
    device = get_device(args.device)
    dataset, loader = create_dataset(cfg, args.split)
    checkpoint = load_checkpoint(args.checkpoint, device)
    check_checkpoint_config(checkpoint, cfg, dataset)
    encoder, decoder = create_models(cfg, dataset, device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()
    predictions = []
    for batch in tqdm(loader, desc='Generate answers'):
        batch = move_batch(batch, device)
        generated = decoder(encoder(batch), mode='sample').cpu().tolist()
        for qa_id, sequence in zip(batch['qa_id'].cpu().tolist(), generated):
            record = dataset.records.iloc[qa_id]
            predictions.append({
                'qa_id': int(qa_id), 'question': str(record['question']),
                'question_type': str(record['question_type']).strip().lower(),
                'reference': str(record['answer']),
                'prediction': decode_tokens(sequence, dataset.idx_to_word, dataset.bos_id),
            })
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'config.json', cfg)
    if not args.predictions_only:
        metrics = evaluate_predictions(predictions, output)
        write_json(output / 'metrics.json', metrics)
        print(metrics)


if __name__ == '__main__':
    main()
