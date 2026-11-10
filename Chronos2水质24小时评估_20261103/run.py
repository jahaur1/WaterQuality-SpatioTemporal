from pathlib import Path
import os,sys,json,time,hashlib,argparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='2'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
REF=HERE.parent/'预训练模型水质24小时评估_20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def make_inputs(h,f,expose):
    result=[]
    for j in range(len(h)):
        row={'target':h[j,:,5:].T.copy(),'past_covariates':{f'easy_{c}':h[j,:,c].copy() for c in range(5)}}
        if expose:
            future={}
            for c in range(5):
                x=np.full(6,np.nan,np.float32);x[-1]=f[j,c];future[f'easy_{c}']=x
            row['future_covariates']=future
        result.append(row)
    return result
def forecast(pipe,h,f,expose):
    # Every origin/window is a distinct task; never share information across origins or stations.
    pp=pipe.predict(make_inputs(h,f,expose),prediction_length=6,batch_size=1152,context_length=32,cross_learning=False)
    q=pipe.quantiles.index(.5)
    assert len(pp)==len(h) and all(p.shape==(4,len(pipe.quantiles),6) for p in pp)
    return np.stack([p[:,q,5].float().cpu().numpy() for p in pp])
def main():
    a=argparse.ArgumentParser();a.add_argument('--limit',type=int,default=0);a=a.parse_args();torch.set_num_threads(2);torch.manual_seed(20261103)
    z=np.load(REF/'inputs.npz');h=z['history'];f=z['future_easy'];N=min(len(h),a.limit) if a.limit else len(h)
    pipe=Chronos2Pipeline.from_pretrained(str(HERE/'weights'),device_map='cuda',torch_dtype=torch.float32,local_files_only=True);pipe.model.eval()
    fingerprint={'weights_sha256':sha(HERE/'weights/model.safetensors'),'inputs_sha256':sha(REF/'inputs.npz'),'limit':a.limit,'N':N,'batch_windows':128,'batch_variates':1152,'cross_learning':False,'precision':'float32','point_quantile':.5,'grid_hours':4,'prediction_length':6,'context_length':32,'fine_tune':False,'chronos_version':__import__('chronos').__version__ if hasattr(__import__('chronos'),'__version__') else '2.3.2'}
    from safetensors import safe_open
    with safe_open(HERE/'weights/model.safetensors',framework='pt') as weights:fingerprint['parameter_tensor_elements']=sum(int(np.prod(weights.get_slice(k).get_shape())) for k in weights.keys())
    for arm,expose in [('history_only',False),('future_easy',True)]:
        dest=HERE/('pilot' if a.limit else 'results')/arm;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'done.json').exists():
            old=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert all(old[k]==v for k,v in fingerprint.items());continue
        raw=np.full((N,4),np.nan,np.float32);done=0;start=time.time()
        if (dest/'partial.npz').exists():
            last=np.load(dest/'partial.npz');raw[:]=last['raw'];done=int(last['done'])
        for lo in range(done,N,128):
            hi=min(lo+128,N);raw[lo:hi]=forecast(pipe,h[lo:hi],f[lo:hi],expose);assert np.isfinite(raw[lo:hi]).all()
            if lo==done or hi==N or (lo//128)%20==0:
                np.savez(dest/'partial.npz',raw=raw,done=hi);dump(dest/'progress.json',{'done':hi,'total':N,'seconds':round(time.time()-start,2)})
                print(arm,hi,'/',N,'seconds',round(time.time()-start,2),flush=True)
        np.savez_compressed(dest/'predictions.npz',raw=raw,prediction=np.maximum(raw,0));dump(dest/'done.json',dict(fingerprint,completed=True,seconds=round(time.time()-start,2)))
    # Fixed-RNG-free quantile inference. Audit native future influence and task isolation.
    base=forecast(pipe,h[:2],f[:2],True);perturb=forecast(pipe,h[:2],f[:2]+10,True)
    hidden=forecast(pipe,h[:2],f[:2],False);hidden2=forecast(pipe,h[:2],f[:2]+10,False)
    isolated=forecast(pipe,h[:1],f[:1],True)
    change=float(np.max(np.abs(base-perturb)));ignored=float(np.max(np.abs(hidden-hidden2)));isolation=float(np.max(np.abs(base[:1]-isolated)))
    assert change>0 and ignored==0 and np.allclose(base[:1],isolated,rtol=1e-4,atol=1e-5)
    dump(HERE/('试跑输入核验.json' if a.limit else '输入隔离核验.json'),{'passed':True,'future_easy_changes_prediction':change,'history_only_ignores_future_easy':ignored,'isolated_task_vs_batch_max_abs_difference':isolation,'no_future_chemistry_in_interface':True,'cross_learning':False,'fingerprint':fingerprint})
if __name__=='__main__':main()
