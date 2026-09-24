"""Load paper settings without import-time data access or device allocation."""
from pathlib import Path

import torch
import yaml

from utils.attr_dict import AttrDict


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MODEL_ROOT / 'configs' / 'dynamic' / 'dynamic_change_pos_mimic.yaml'


def _as_attr(value):
    if isinstance(value, dict):
        return AttrDict({key: _as_attr(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_as_attr(item) for item in value]
    return value


def load_config(filename=None, data_dir=None, bert_dir=None):
    filename = Path(filename or DEFAULT_CONFIG).expanduser().resolve()
    with filename.open(encoding='utf-8') as stream:
        config = _as_attr(yaml.safe_load(stream))
    for name, override in [('data_dir', data_dir), ('bert_dir', bert_dir)]:
        path = Path(override or config.paths[name]).expanduser()
        # CLI relative paths are relative to cwd; YAML defaults to model/.
        if not path.is_absolute():
            path = (Path.cwd() if override else MODEL_ROOT) / path
        config.paths[name] = str(path.resolve())
    if config.model.dsmv.num_layers < 1:
        raise ValueError('num_layers must be positive.')
    for key in ('image_heads', 'fusion_heads'):
        if config.model.dsmv.d_model % config.model.dsmv[key]:
            raise ValueError('d_model must be divisible by attention heads.')
    return config


def get_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(requested)
