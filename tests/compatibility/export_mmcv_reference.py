"""Run with an existing MMCV environment; writes only the requested fixture.

Example: CUDA_VISIBLE_DEVICES=0 OLD_PYTHON tests/compatibility/export_mmcv_reference.py /tmp/mmcv.pt
"""
import sys
import torch
import mmcv
from mmcv.ops import DeformConv2d

torch.set_num_threads(1)
torch.manual_seed(317)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cases = []
for groups, stride, dilation, zero in [(1,1,1,False), (2,1,1,False), (1,2,1,False), (1,1,2,False), (1,1,1,True)]:
    op = DeformConv2d(8, 8, 3, padding=dilation, stride=stride, dilation=dilation,
                      deform_groups=8, groups=groups, bias=False, im2col_step=128).to(device)
    x = torch.randn(2,8,6,8,device=device,requires_grad=True)
    oh, ow = (6-1)//stride+1, (8-1)//stride+1
    offset = (torch.rand(2,144,oh,ow,device=device)*1.2-0.6)
    if zero: offset.zero_()
    offset.requires_grad_()
    y = op(x,offset)
    probe = torch.randn_like(y)
    (y*probe).sum().backward()
    cases.append(dict(groups=groups,stride=stride,dilation=dilation,zero=zero,
        x=x.detach().cpu(),offset=offset.detach().cpu(),weight=op.weight.detach().cpu(),
        output=y.detach().cpu(),probe=probe.cpu(),dx=x.grad.cpu(),do=offset.grad.cpu(),dw=op.weight.grad.cpu()))
torch.save(dict(torch_version=torch.__version__,mmcv_version=mmcv.__version__,device=device,cases=cases),sys.argv[1])
print('Exported MMCV reference:',torch.__version__,mmcv.__version__,device,len(cases))
