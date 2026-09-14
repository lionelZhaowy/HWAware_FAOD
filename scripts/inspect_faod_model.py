"""Print a CPU shape trace of the default FAOD model, without training or data files.

Example: python -B scripts/inspect_faod_model.py --dataset pku_fusion --output /tmp/faod.json
Uses random initialization and synthetic inputs: this is a structure inspection,
not an accuracy or throughput benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['pku_fusion', 'dsec'], default='pku_fusion')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    from hydra import compose, initialize_config_dir
    from config.modifier import dynamically_modify_train_config
    from modules.detection_fusion import Module

    torch.set_num_threads(1)
    torch.manual_seed(317)
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base='1.2'):
        cfg = compose(config_name='val', overrides=[f'dataset={args.dataset}',
                       f'+experiment/{args.dataset}=base.yaml'])
    dynamically_modify_train_config(cfg)
    module = Module(cfg).eval()
    model = module.mdl
    trace = {}

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        return value if isinstance(value, (int, float, str, bool)) else None

    names = [f'backbone.stages.{i}' for i in range(4)] + [
        'backbone.stages.0.ev_initial_layers', 'backbone.stages.0.img_initial_layers',
        'backbone.stages.0.align_block', 'backbone.stages.0.align_block.guide',
        'backbone.stages.0.align_block.align.dconv_1.conv_offset',
        'backbone.stages.0.align_block.align.dconv_1.dconv',
        'backbone.stages.0.fusion_block', 'fpn', 'yolox_head']
    modules = dict(model.named_modules())
    handles = []
    for name in names:
        def hook(layer, inputs, outputs, key=name):
            trace[key] = {'type': type(layer).__module__ + '.' + type(layer).__name__,
                          'inputs': describe(inputs), 'outputs': describe(outputs)}
        handles.append(modules[name].register_forward_hook(hook))
    height, width = cfg.dataset.resolution_hw
    event = module.input_padder.pad_tensor_repr(torch.rand(1,20,height,width))
    frame = module.input_padder.pad_tensor_repr(torch.rand(1,3,height,width))
    with torch.inference_mode():
        features, states = model.forward_backbone_rnn(event, frame)
        predictions, _ = model.forward_detect(features)
    for handle in handles:
        handle.remove()
    assert torch.isfinite(predictions).all()
    result = {
        'dataset': args.dataset, 'inspection_only': True, 'weights': 'random initialization',
        'device': 'cpu', 'native_hw': [height, width], 'padded_event': describe(event),
        'padded_frame': describe(frame), 'stage_channels': model.backbone.stage_dims,
        'stage_strides': model.backbone.strides, 'feature_shapes': describe(features),
        'state_shapes': describe(states), 'prediction': describe(predictions),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'parameter_breakdown': {
            **{f'stage{i+1}': sum(p.numel() for p in stage.parameters())
               for i, stage in enumerate(model.backbone.stages)},
            'fpn': sum(p.numel() for p in model.fpn.parameters()),
            'head': sum(p.numel() for p in model.yolox_head.parameters())},
        'trace': trace}
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n')
        print('TRACE_SAVED', args.output)
    else:
        print(output)
    print('SUMMARY', args.dataset, result['parameter_count'], result['prediction'])


if __name__ == '__main__':
    main()
