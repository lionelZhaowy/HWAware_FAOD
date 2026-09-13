"""Original FAOD synthetic training/restore/inference smoke test, no dataset writes.

Run: python -m tests.compatibility.smoke_training is not required; direct script works.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from torch.utils.data import Dataset, DataLoader
from data.ev_img_dataloader.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.types import DataType
from loggers.wandb_logger import WandbLogger
from modules.detection_fusion import Module
from config.modifier import dynamically_modify_train_config


class Clips(Dataset):
    def __len__(self): return 2
    def __getitem__(self,index):
        g=torch.Generator().manual_seed(317+index)
        labels=ObjectLabels(torch.tensor([[8333.,20.,12.,20.,16.,0.]]),(64,96))
        return {'worker_id':0,'data':{
            DataType.EV_REPR:[torch.rand(1,20,64,96,generator=g) for _ in range(2)],
            DataType.IMAGE:[torch.rand(1,3,64,96,generator=g) for _ in range(2)],
            DataType.OBJLABELS_SEQ:[SparselyBatchedObjectLabels([None]),SparselyBatchedObjectLabels([labels])],
            DataType.IS_FIRST_SAMPLE:torch.tensor([True]),
            DataType.IS_PADDED_MASK:[False,False],DataType.DRIFT:torch.tensor([0])}}


class Record(pl.Callback):
    def __init__(self): self.losses=[]; self.finite_gradient_steps=0
    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        if all(torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group["params"] if p.grad is not None):
            self.finite_gradient_steps+=1
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        loss=float(outputs['loss'])
        assert torch.isfinite(torch.tensor(loss))
        self.losses.append(loss)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--precision',default='32-true')
    args=p.parse_args();torch.set_num_threads(1);pl.seed_everything(317,workers=True)
    with initialize_config_dir(config_dir=str(ROOT/'config'),version_base='1.2'):
        cfg=compose(config_name='train',overrides=['dataset=dsec','+experiment/dsec=base.yaml'])
    dynamically_modify_train_config(cfg)
    cfg.model.backbone.in_res_hw=[64,96]
    cfg.model.backbone.stage.attention.partition_size=[2,3]
    cfg.dataset.train.sampling='random';cfg.training.lr_scheduler.use=False
    cfg.logging.train.metrics.compute=False
    initial_steps=12 if args.precision=='16-mixed' else 2
    accelerator='gpu' if args.device.startswith('cuda') else 'cpu'
    devices=[int(args.device.split(':')[-1])] if accelerator=='gpu' else 1
    with tempfile.TemporaryDirectory(prefix='faod-smoke-') as output:
        logger=WandbLogger(project='faod-environment-smoke',mode='disabled',log_model=False,dir=output)
        recorder=Record()
        kwargs=dict(accelerator=accelerator,devices=devices,precision=args.precision,strategy='auto',
                    default_root_dir=output,logger=logger,enable_checkpointing=False,enable_progress_bar=False,
                    enable_model_summary=False,limit_val_batches=0,num_sanity_val_steps=0,log_every_n_steps=1,
                    gradient_clip_val=1.,gradient_clip_algorithm='value',callbacks=[recorder])
        model=Module(cfg)
        print('ACTUAL MODEL:',type(model.mdl.backbone).__module__,type(model.mdl.backbone).__name__)
        print('PARAMETERS:',sum(x.numel() for x in model.parameters()))
        trainer=pl.Trainer(max_steps=initial_steps,**kwargs)
        trainer.fit(model,train_dataloaders=DataLoader(Clips(),batch_size=None,num_workers=0))
        assert trainer.global_step==initial_steps
        checkpoint=Path(output)/'resume.ckpt';trainer.save_checkpoint(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['global_step']==initial_steps and saved['optimizer_states'][0]['state']
        assert recorder.finite_gradient_steps>0
        restored=Module(cfg);restored.load_state_dict(saved['state_dict'],strict=True)
        restored_trainer=pl.Trainer(max_steps=initial_steps+1,**kwargs)
        restored_trainer.fit(restored,train_dataloaders=DataLoader(Clips(),batch_size=None,num_workers=0),ckpt_path=checkpoint)
        assert restored_trainer.global_step==initial_steps+1
        restored.eval();device=restored.device
        with torch.no_grad():
            features,states=restored.mdl.forward_backbone_rnn(torch.rand(1,20,64,96,device=device),
                                                             torch.rand(1,3,64,96,device=device))
            pred,loss=restored.mdl.forward_detect(features)
        assert loss is None and torch.isfinite(pred).all()
        print('RESULT',json.dumps(dict(train_device=args.device,inference_device=str(device),precision=args.precision,steps=restored_trainer.global_step,
              losses=recorder.losses,finite_gradient_steps=recorder.finite_gradient_steps,prediction_shape=list(pred.shape),checkpoint_restore=True,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)) if accelerator=='gpu' else None)))
    import wandb
    wandb.finish()


if __name__=='__main__': main()
