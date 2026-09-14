"""Run the real train.py CLI, additionally saving initial parameters for an audit.

Set FAOD_INITIAL_SNAPSHOT to a new local .pt path, and pass ordinary train.py
Hydra overrides. This wrapper does not change the Trainer, model, loss, optimizer,
sampling, or random seed; it captures parameters immediately after construction.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train as entry
import torch
import hydra


def main():
    snapshot = Path(os.environ['FAOD_INITIAL_SNAPSHOT']).expanduser().resolve()
    if snapshot.exists():
        raise FileExistsError(f'Refusing to overwrite initial snapshot: {snapshot}')
    factory = entry.fetch_model_module

    def capture_initial(config):
        module = factory(config)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        parameters = {name: value.detach().cpu().clone()
                      for name, value in module.named_parameters()}
        torch.save(parameters, snapshot)
        print(f'INITIAL_PARAMETERS_SAVED {snapshot} ({len(parameters)} tensors)', flush=True)
        return module

    entry.fetch_model_module = capture_initial
    try:
        # Calling an imported Hydra entry otherwise searches for a config package.
        # Pin the same on-disk config directory used by the original train.py CLI.
        @hydra.main(config_path=str(ROOT / "config"), config_name="train", version_base="1.2")
        def run(config):
            return entry.main.__wrapped__(config)
        run()
    finally:
        entry.fetch_model_module = factory


if __name__ == '__main__':
    main()
