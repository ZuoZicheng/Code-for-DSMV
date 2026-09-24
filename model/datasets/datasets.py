import torch
from torch.utils.data import DataLoader

from datasets.rcc_dataset_pos_mimic import RCCDataset_mimic


def create_dataset(cfg, split='train'):
    if cfg.data.dataset != 'rcc_dataset_mimic':
        raise ValueError('Only the prepared MIMIC-Diff-VQA dataset is supported.')
    dataset = RCCDataset_mimic(cfg, split)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        dataset, batch_size=dataset.batch_size, shuffle=(split == 'train'),
        num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return dataset, loader
