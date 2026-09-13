"""Behavior contracts for the environment/API upgrade (no real data or W&B network)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, MapDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper

from data.utils.datapipes import MapToIter, Repeat, ZipperLongest
from data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from data.utils.stream_sharded_datapipe import ShardedStreamingDataPipe
from models.layers.compat.deform_conv import DeformConv2d


class Sequence(MapDataPipe):
    def __init__(self, identity, length): self.identity, self.length = identity, length
    def __len__(self): return self.length
    def __getitem__(self, index):
        if not 0 <= index < self.length: raise IndexError(index)
        return self.identity, index


def stream_results(concat, workers):
    torch.manual_seed(317)
    sequences = [Sequence(i, 8-i) for i in range(6)]
    if concat:
        stream = ConcatStreamingDataPipe(sequences, batch_size=2, num_workers=workers)
    else:
        stream = ShardedStreamingDataPipe(sequences, batch_size=2, fill_value=None)
    loader = DataLoader(stream, batch_size=None, num_workers=workers,
                        generator=torch.Generator().manual_seed(123))
    return list(loader)


def test_stream_adapters():
    assert list(MapToIter(Sequence(7,2))) == [(7,0),(7,1)]
    assert list(Repeat(IterableWrapper([1,2]),2)) == [1,2,1,2]
    assert list(Repeat(IterableWrapper([]))) == []
    assert list(ZipperLongest(IterableWrapper([1,2]),IterableWrapper([3]))) == [(1,3),(2,None)]


@pytest.mark.parametrize('workers', [0,2])
def test_stream_no_loss_or_duplication(workers):
    batches = stream_results(False,workers)
    samples = [tuple(x) for batch, worker in batches for x in batch if x is not None]
    assert sorted(samples) == [(i,j) for i in range(6) for j in range(8-i)]


def test_rank_worker_ownership():
    sequences = [Sequence(i, 8-i) for i in range(6)]
    assigned = [ShardedStreamingDataPipe.assign_datapipes_to_worker(sequences,4,r) for r in range(4)]
    assert sorted(s.identity for group in assigned for s in group) == list(range(6))


@pytest.mark.skipif(not os.getenv('FAOD_STREAM_REFERENCE'), reason='optional original TorchData fixture')
def test_original_stream_equivalence():
    expected = json.loads(Path(os.environ['FAOD_STREAM_REFERENCE']).read_text())
    for concat in (False,True):
        for workers in (0,2):
            actual = json.loads(json.dumps(stream_results(concat,workers)))
            assert actual == expected[f'{concat}_{workers}']


def deform_reference(x, offset, weight, groups, deform_groups, stride, padding, dilation):
    """Independent bilinear sampling reference with MMCV's (dy,dx) ordering."""
    b,c,h,w = x.shape
    cout, _, kh, kw = weight.shape
    oh,ow = offset.shape[-2:]
    yy,xx = torch.meshgrid(torch.arange(oh,device=x.device,dtype=x.dtype),
                          torch.arange(ow,device=x.device,dtype=x.dtype),indexing='ij')
    samples = []
    for ky in range(kh):
        for kx in range(kw):
            by_group = []
            for g in range(deform_groups):
                slot = g*kh*kw+ky*kw+kx
                y = yy*stride-padding+ky*dilation+offset[:,2*slot]
                z = xx*stride-padding+kx*dilation+offset[:,2*slot+1]
                grid = torch.stack((2*z/(w-1)-1,2*y/(h-1)-1),-1)
                part = x[:,g*(c//deform_groups):(g+1)*(c//deform_groups)]
                by_group.append(F.grid_sample(part,grid,align_corners=True,padding_mode='zeros'))
            samples.append(torch.cat(by_group,1))
    samples = torch.stack(samples,2)
    outputs=[]
    for g in range(groups):
        part=samples[:,g*(c//groups):(g+1)*(c//groups)]
        wg=weight[g*(cout//groups):(g+1)*(cout//groups)].flatten(2)
        outputs.append(torch.einsum('bckhw,ock->bohw',part,wg))
    return torch.cat(outputs,1)


@pytest.mark.parametrize('groups,stride,dilation',[(1,1,1),(2,1,1),(1,2,2)])
def test_deform_output_and_gradient_reference(groups,stride,dilation):
    torch.manual_seed(41)
    op=DeformConv2d(8,8,3,groups=groups,deform_groups=8,stride=stride,padding=dilation,dilation=dilation).double()
    x=torch.randn(2,8,5,7,dtype=torch.double,requires_grad=True)
    offset=(torch.rand(2,144,(5-1)//stride+1,(7-1)//stride+1,dtype=torch.double)*1.2-0.6).requires_grad_()
    y=op(x,offset)
    ref=deform_reference(x,offset,op.weight,groups,8,stride,dilation,dilation)
    torch.testing.assert_close(y,ref,rtol=1e-9,atol=1e-9)
    probe=torch.randn_like(y)
    grad=torch.autograd.grad((y*probe).sum(),(x,offset,op.weight),retain_graph=True)
    ref_grad=torch.autograd.grad((ref*probe).sum(),(x,offset,op.weight))
    for a,b in zip(grad,ref_grad): torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-8)


def test_zero_offset_and_checkpoint_contract():
    torch.manual_seed(317)
    op=DeformConv2d(8,8,3,padding=1,deform_groups=8)
    torch.manual_seed(317)
    expected=torch.empty_like(op.weight)
    torch.nn.init.kaiming_uniform_(expected,nonlinearity='relu')
    assert list(op.state_dict()) == ['weight']
    torch.testing.assert_close(op.weight,expected,rtol=0,atol=0)
    op.load_state_dict({'weight':expected},strict=True)
    x=torch.randn(2,8,5,7)
    torch.testing.assert_close(op(x,torch.zeros(2,144,5,7)),F.conv2d(x,op.weight,padding=1))


@pytest.mark.skipif(not os.getenv('FAOD_MMCV_REFERENCE'), reason='optional independently exported MMCV fixture')
def test_compiled_mmcv_equivalence():
    fixture=torch.load(os.environ['FAOD_MMCV_REFERENCE'],weights_only=False,map_location='cpu')
    device=os.getenv('FAOD_TEST_DEVICE','cpu')
    errors={k:0. for k in ('output','dx','do','dw')}
    for case in fixture['cases']:
        op=DeformConv2d(8,8,3,groups=case['groups'],deform_groups=8,stride=case['stride'],
                       padding=case['dilation'],dilation=case['dilation']).to(device)
        op.load_state_dict({'weight':case['weight']})
        x=case['x'].to(device).requires_grad_(); offset=case['offset'].to(device).requires_grad_()
        y=op(x,offset);(y*case['probe'].to(device)).sum().backward()
        for key,actual in [('output',y),('dx',x.grad),('do',offset.grad),('dw',op.weight.grad)]:
            actual=actual.detach().cpu();expected=case[key]
            # Offset derivative at an integer/border is not unique: compare nonzero cases for do.
            if key=='do' and case['zero']: continue
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
            errors[key]=max(errors[key],float((actual-expected).abs().max()))
    print('MMCV comparison max absolute errors:',errors)


def test_logger_disabled_and_pickle(tmp_path):
    import pickle,wandb
    from loggers.wandb_logger import WandbLogger
    wandb.finish()
    with patch('wandb.Api',side_effect=AssertionError('No external API allowed')):
        logger=WandbLogger(project='faod-compatibility',mode='disabled',log_model=False,dir=str(tmp_path),
                           config_args={'test':True})
        logger.log_hyperparams({'nested':{'lr':0.01}})
        logger.log_metrics({'loss':torch.tensor(1.)},step=1)
        logger.log_images('image',[torch.zeros(3,4,4).numpy()],step=1)
        assert logger.experiment.config['nested/lr']==0.01
        assert logger.__getstate__()['_experiment'] is None
        restored=pickle.loads(pickle.dumps(logger))
        assert restored._experiment is None
        assert restored.experiment is logger.experiment
        logger.finalize('success')
    wandb.finish()


def test_checkpoint_artifact_offline_policy(tmp_path):
    from loggers.wandb_logger import WandbLogger
    logger=object.__new__(WandbLogger)
    file=tmp_path/'best.ckpt';file.write_bytes(b'fixture')
    from wandb.sdk.wandb_run import Run
    logger._experiment=Mock(spec=Run)
    logger._experiment.id='test-run'
    logger._experiment.settings=SimpleNamespace(mode='offline')
    logger._checkpoint_callback=None;logger._logged_model_time={};logger._log_model=True
    callback=SimpleNamespace(best_k_models={str(file):torch.tensor(0.8)},last_model_path='',
        current_score=None,best_model_score=torch.tensor(0.8),best_model_path=str(file),monitor='AP',mode='max',save_last=False,
        save_top_k=1,save_weights_only=False)
    with patch.object(logger,'_num_logged_artifact',side_effect=AssertionError('No network')):
        logger._scan_and_log_checkpoints(callback,False)
        assert logger._experiment.log_artifact.call_count==1
        assert logger._experiment.log_artifact.call_args.kwargs['aliases']==['best']
        logger._scan_and_log_checkpoints(callback,False)
        assert logger._experiment.log_artifact.call_count==1


def test_yolox_grid_cache_device_dtype_and_shape():
    from models.detection.yolox.models.yolo_head import YOLOXHead
    head=YOLOXHead(num_classes=2,in_channels=(8,16,32))
    keys=set(head.state_dict())
    devices=['cpu','cuda:0','cpu'] if torch.cuda.is_available() else ['cpu']
    for device in devices:
        for dtype in (torch.float32,torch.float64):
            head.to(device=device,dtype=dtype)
            for hw in (((4,6),(2,3),(1,2)),((6,4),(3,2),(2,1))):
                head.hw=hw
                expected=[]
                for (height,width),stride in zip(hw,head.strides):
                    for y in range(height):
                        for x in range(width):
                            expected.append([x*stride,y*stride,stride,stride,0,0,0])
                raw=torch.zeros(1,len(expected),7,device=device,dtype=dtype)
                torch.testing.assert_close(head.decode_outputs(raw),torch.tensor([expected],device=device,dtype=dtype))
                train_raw=torch.zeros(1,7,*hw[0],device=device,dtype=dtype)
                decoded,grid=head.get_output_and_grid(train_raw,0,8,train_raw.type())
                torch.testing.assert_close(decoded,torch.tensor([expected[:hw[0][0]*hw[0][1]]],device=device,dtype=dtype))
                assert grid.device==raw.device and grid.dtype==dtype
    assert set(head.state_dict())==keys
