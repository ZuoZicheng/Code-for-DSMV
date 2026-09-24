"""Question-independent knowledge encoding followed by the DSMV encoder."""
import torch
from torch import nn

from models.relation_encoder import ExplicitRelationEncoder, ImplicitRelationEncoder
from models.transformer import TransformerEncoder
from utils.mimic_utils import (
    process_matrix, torch_extract_position_matrix, torch_extract_position_embedding,
)


class ChangeDetector(nn.Module):
    def __init__(self, cfg, word_to_idx):
        super().__init__()
        self.cfg = cfg
        options = cfg.model.change_detector
        self.img = nn.Linear(options.feat_dim, options.att_dim)
        if options.att_dim != cfg.model.dsmv.d_model:
            raise ValueError('Graph output and DSMV widths must match.')
        self.coef_sem = options.coef_sem
        self.coef_spa = options.coef_spa
        if min(self.coef_sem, self.coef_spa) < 0 or self.coef_sem + self.coef_spa > 1:
            raise ValueError('Graph coefficients must form nonnegative mixture weights.')
        # Preserve EKAID relation types, but remove question injection before stage 1.
        shared = dict(v_dim=options.att_dim, q_dim=0, out_dim=options.att_dim,
                      dir_num=options.dir_num, nongt_dim=options.nongt_dim,
                      num_heads=options.att_head, num_steps=1,
                      residual_connection=True, label_bias=False)
        self.semantic_relation = ExplicitRelationEncoder(
            label_num=options.sem_label_num, **shared)
        self.spatial_relation = ExplicitRelationEncoder(
            label_num=options.spa_label_num, **shared)
        self.implicit_relation = ImplicitRelationEncoder(
            pos_emb_dim=options.pos_emb_dim, **shared)
        self.encoder = TransformerEncoder(cfg, word_to_idx)

    def encode_image(self, features, spatial, semantic, boxes):
        options = self.cfg.model.change_detector
        if features.size(1) != options.nongt_dim:
            raise ValueError('The full DSMV model expects 52 visual tokens per image.')
        image = self.img(features.float())
        spatial = process_matrix(spatial, self.cfg, image.size(1), image.device, 'spatial')
        semantic = process_matrix(semantic, self.cfg, image.size(1), image.device, 'semantic')
        positions = torch_extract_position_matrix(boxes.float(), options.nongt_dim)
        positions = torch_extract_position_embedding(
            positions, options.pos_emb_dim, device=image.device)
        sem, _ = self.semantic_relation(image, semantic)
        spa, _ = self.spatial_relation(image, spatial)
        imp, _ = self.implicit_relation(image, positions)
        return self.coef_sem * sem + self.coef_spa * spa + (1 - self.coef_sem - self.coef_spa) * imp

    def forward(self, batch):
        before = self.encode_image(batch['before'], batch['before_spatial'],
                                   batch['before_semantic'], batch['before_boxes'])
        after = self.encode_image(batch['after'], batch['after_spatial'],
                                  batch['after_semantic'], batch['after_boxes'])
        return self.encoder(before, after, batch['question'])
