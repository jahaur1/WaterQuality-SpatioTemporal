"""Verify native covariate use and absence of future chemical data in model calls."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,json,hashlib,importlib.metadata
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from uni2ts.model.moirai import MoiraiModule,MoiraiForecast
HERE=Path(__file__).resolve().parent;torch.set_num_threads(2)
z=np.load(HERE/'inputs.npz');h=z['history'][:8];f=z['future_easy'][:8];valid=np.isfinite(h);bs=len(h)
module=MoiraiModule.from_pretrained('D:/AI/TATO-main/CKPT/MOIRAI-small',local_files_only=True)
model=MoiraiForecast(module=module,prediction_length=6,target_dim=4,feat_dynamic_real_dim=5,past_feat_dynamic_real_dim=0,context_length=32,patch_size=8,num_samples=20).to('cuda').eval()
def predict(expose,change):
    dyn=np.concatenate([np.nan_to_num(h[:,:,:5]),np.zeros((bs,6,5),np.float32)],axis=1);mask=np.concatenate([valid[:,:,:5],np.zeros((bs,6,5),bool)],axis=1)
    if expose:dyn[:,-1]=f+change;mask[:,-1]=True
    kwargs={'past_target':torch.tensor(np.nan_to_num(h[:,:,5:]),device='cuda'),'past_observed_target':torch.tensor(valid[:,:,5:],device='cuda'),'past_is_pad':torch.tensor(~valid.any(-1),device='cuda'),'feat_dynamic_real':torch.tensor(dyn,dtype=torch.float32,device='cuda'),'observed_feat_dynamic_real':torch.tensor(mask,device='cuda')}
    assert kwargs['past_target'].shape==(bs,32,4) and kwargs['feat_dynamic_real'].shape==(bs,38,5)
    # The interface contains no future target array and only exactly 5 future covariates.
    assert 'future_target' not in kwargs and 'future_observed_target' not in kwargs
    torch.manual_seed(98451)
    with torch.inference_mode():return model(**kwargs).cpu().numpy()
base=predict(True,0);changed=predict(True,10);hidden=predict(False,0);hidden_again=predict(False,10)
effect=float(np.max(np.abs(base-changed)));hidden_effect=float(np.max(np.abs(hidden-hidden_again)));assert effect>0 and hidden_effect==0
import uni2ts.model.moirai.forecast as fc
import uni2ts.model.moirai.module as md
hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [fc.__file__,md.__file__]}
out={'passed':True,'native_future_easy_perturbation_changes_output_with_identical_rng':effect,'history_only_ignores_target_time_easy_with_identical_rng':hidden_effect,'future_chemical_target_not_in_model_interface':True,'future_easy_channels':5,'history_chemical_channels':4,'future_horizon_steps':6,'grid_hours':4,'uni2ts_version':importlib.metadata.version('uni2ts'),'native_source_sha256':hashes}
(HERE/'未来输入隔离核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
