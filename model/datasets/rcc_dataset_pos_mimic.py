"""Read prepared MIMIC-Diff-VQA data; never generate or overwrite splits."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RCCDataset_mimic(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        root = Path(cfg.paths.data_dir)
        with (root / cfg.data.vocab_json).open(encoding='utf-8') as stream:
            self.word_to_idx = json.load(stream)
        self.idx_to_word = {int(index): word for word, index in self.word_to_idx.items()}
        self.vocab_size = max(self.idx_to_word) + 1
        if '<start>' not in self.word_to_idx:
            raise ValueError('Expected <start> in the EKAID answer vocabulary.')
        self.bos_id = int(self.word_to_idx['<start>'])
        if self.bos_id == 0:
            raise ValueError('The prepared data reserve 0 for EOS/padding.')
        with (root / cfg.data.splits_json).open(encoding='utf-8') as stream:
            splits = json.load(stream)
        self.split_idxs = [int(index) for index in splits[split]]
        self.batch_size = cfg.data[split].batch_size
        with h5py.File(root / cfg.data.h5_label_file, 'r') as labels:
            self.answers = labels['answers'][:].astype(np.int64)
            self.questions = labels['questions'][:].astype(np.int64)
            self.feature_idx = labels['feature_idx'][:].astype(np.int64)
        # CSV row numbers are the QA identifiers used by dataset_preparation.py.
        self.records = pd.read_csv(root / cfg.data.questions_csv, keep_default_na=False)
        required = {'question', 'answer', 'question_type'}
        if not required.issubset(self.records.columns):
            raise ValueError('Question CSV must contain question, answer and question_type.')
        if not (len(self.records) == len(self.answers) == len(self.questions) == len(self.feature_idx)):
            raise ValueError('CSV and prepared HDF5 must have the same QA row indexing.')
        if cfg.data.pair_order != 'main_reference':
            raise ValueError('Expected feature_idx columns [main/after, reference/before].')
        self.feature_path = str(root / cfg.data.h5_feature_file)
        self._features = None
        self._owner_pid = None

    def _feature_store(self):
        # Each DataLoader process opens its own read-only HDF5 handle.
        if self._features is None or self._owner_pid != os.getpid():
            if self._features is not None:
                self._features.close()
            self._features = h5py.File(self.feature_path, 'r')
            self._owner_pid = os.getpid()
        return self._features

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_features'] = None
        state['_owner_pid'] = None
        return state

    def __len__(self):
        return len(self.split_idxs)

    def __getitem__(self, index):
        qa_id = self.split_idxs[index]
        store = self._feature_store()
        # Source dataset_preparation.py explicitly saves [study_id, ref_id].
        after_id, before_id = self.feature_idx[qa_id]
        item = {'qa_id': qa_id, 'question': torch.from_numpy(self.questions[qa_id].copy())}
        for role, feature_id in [('before', before_id), ('after', after_id)]:
            item[role] = torch.as_tensor(store['image_features'][feature_id], dtype=torch.float32)
            item[role + '_boxes'] = torch.as_tensor(store['image_bb'][feature_id], dtype=torch.float32)
            item[role + '_spatial'] = torch.as_tensor(store['image_adj_matrix'][feature_id], dtype=torch.long)
            item[role + '_semantic'] = torch.as_tensor(store['semantic_adj_matrix'][feature_id], dtype=torch.long)
        # One QA row supplies its own question, answer and image pair; no random row offset.
        steps = self.cfg.model.speaker.seq_length
        row = self.answers[qa_id]
        if row[0] != self.bos_id:
            raise ValueError('Answer rows must begin with the vocabulary <start> token.')
        sequence = np.zeros(steps + 1, dtype=np.int64)
        copy_length = min(len(row), steps + 1)
        sequence[:copy_length] = row[:copy_length]
        targets = sequence[1:]
        # Prepared EKAID data use 0 for both EOS and later padding. Supervise
        # the first zero as EOS, exclude every padding position after it.
        zeros = np.flatnonzero(targets == 0)
        valid_length = int(zeros[0] + 1) if len(zeros) else steps
        mask = np.zeros(steps, dtype=np.float32)
        mask[:valid_length] = 1
        item['input_tokens'] = torch.from_numpy(sequence[:-1].copy())
        item['targets'] = torch.from_numpy(targets.copy())
        item['mask'] = torch.from_numpy(mask)
        return item

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_vocab_size(self):
        return self.vocab_size
