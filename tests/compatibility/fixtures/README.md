# Legacy stream fixture

`legacy_stream.json` was exported on 2026-09-13 from unmodified FAOD
`e8666ca536850807173502e6764423135194cf7f` implementations of
`data/utils/stream_concat_datapipe.py` and `stream_sharded_datapipe.py`, using
PyTorch 2.5.0 and a temporary TorchData 0.9.0 installation outside the shared
Conda environment. TorchData was not installed into that environment.

The test `stream_results` documents the input: six map-style sequences of lengths
8, 7, 6, 5, 4, 3; batch size 2; torch seed 317; DataLoader generator seed 123;
0 or 2 workers. Keys are `<concat>_<workers>`. Each value includes every emitted
batch and its worker ID, including padding and repetition in the original concat
pipeline. Counts: False_0=17, False_2=18, True_0=33, True_2=66.

The independent MMCV fixture is generated separately with
`../export_mmcv_reference.py` in an existing MMCV environment. It is not required
for routine local tests; the double-precision grid-sample reference test also
runs without MMCV.
