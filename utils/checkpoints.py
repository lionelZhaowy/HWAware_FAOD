"""Resolve checkpoint overrides against the repository, independent of Hydra chdir."""
from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_checkpoint_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {path}. Download the matching README checkpoint '
            'to checkpoints/ or pass checkpoint=/absolute/path/model.ckpt.'
        )
    return path
