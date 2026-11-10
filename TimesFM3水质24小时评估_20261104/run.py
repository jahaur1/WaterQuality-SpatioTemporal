"""Frozen official TimesFM3 decode with native missing-value masks, no interpolation."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,subprocess
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'source/src'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='2'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from timesfm3 import TimesFM3Forecaster
REF=HERE.parent/'预训练模型水质24小时评估_20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def forecast(model,h,f,expose,masked_fill=0.):
    # Both arms retain the same native covariate role, differing only in observed future values.
    b=len(h);valid=np.isfinite(h)
    target=h[:,:,5:].transpose(0,2,1).copy();tm=~valid[:,:,5:].transpose(0,2,1)
    pf=np.full((b,5,38),np.nan,np.float32);pf[:,:,:32]=h[:,:,:5].transpose(0,2,1)
    if expose:pf[:,:,-1]=f
    fm=~np.isfinite(pf)
    target=np.where(tm,masked_fill,target);pf=np.where(fm,masked_fill,pf)
    kw={'target':torch.tensor(target,dtype=torch.float32,device='cuda'),'horizon':6,'target_mask':torch.tensor(tm,device='cuda'),'past_future_covariates':torch.tensor(pf,dtype=torch.float32,device='cuda'),'past_future_mask':torch.tensor(fm,device='cuda'),'mask':torch.tensor(~valid.any(-1),device='cuda')}
    assert kw['target'].shape==(b,4,32) and kw['past_future_covariates'].shape==(b,5,38)
    # No chemical future values appear in the interface; their horizon is generated inside decode.
    with torch.inference_mode():out=model.decode(**kw)
    assert out.shape==(b,9,6,9),out.shape
    out=out[:,:4,5,:].sort(dim=-1).values[:,:,4]
    result=out.float().cpu().numpy();assert np.isfinite(result).all();return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(20261104)
    z=np.load(REF/'inputs.npz');h=z['history'];f=z['future_easy'];N=min(len(h),a.limit) if a.limit else len(h)
    audit=json.loads((HERE/'下载SHA256核验.json').read_text(encoding='utf-8'));assert audit['passed'] and sha(HERE/'weights/model.safetensors')==audit['sha256']
    pipe=TimesFM3Forecaster.from_pretrained(str(HERE/'weights'),device='cuda',local_files_only=True);model=pipe.model.eval()
    for p in model.parameters():p.requires_grad_(False)
    git_sha=subprocess.check_output(['git','-c','safe.directory='+str((HERE/'source').as_posix()),'rev-parse','HEAD'],cwd=HERE/'source',text=True).strip()
    from safetensors import safe_open
    with safe_open(HERE/'weights/model.safetensors',framework='pt') as w:count=sum(int(np.prod(w.get_slice(k).get_shape())) for k in w.keys())
    fingerprint={'weights_sha256':audit['sha256'],'inputs_sha256':sha(REF/'inputs.npz'),'limit':a.limit,'N':N,'batch_windows':64,'precision':'float32','point_quantile':.5,'sort_quantiles':True,'grid_hours':4,'prediction_length':6,'context_length':32,'fine_tune':False,'official_source_commit':git_sha,'parameter_tensor_elements':count,'native_api':'TimesFM3Torch.decode with explicit target_mask/past_future_mask','history_covariate_role':'same past_future_covariates in both arms; all future masked in history_only','high_level_wrapper_interpolation_used':False,'symmetric_averaging':False,'torch':torch.__version__}
    # Record actual resources (the bundled skill preflight only estimates the older 2.5 model).
    import shutil,psutil
    dump(HERE/'环境核验.json',{'RAM_available_GB':psutil.virtual_memory().available/1e9,'disk_free_GB':shutil.disk_usage(HERE).free/1e9,'GPU':torch.cuda.get_device_name(0),'VRAM_GB':torch.cuda.get_device_properties(0).total_memory/1e9,'weights_GB':(HERE/'weights/model.safetensors').stat().st_size/1e9,'fingerprint':fingerprint})
    for arm,expose in [('history_only',False),('future_easy',True)]:
        dest=HERE/('pilot' if a.limit else 'results')/arm;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'done.json').exists():
            old=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert all(old[k]==v for k,v in fingerprint.items());continue
        raw=np.full((N,4),np.nan,np.float32);done=0;start=time.time()
        if (dest/'partial.npz').exists():
            old=np.load(dest/'partial.npz');raw[:]=old['raw'];done=int(old['done'])
        for lo in range(done,N,64):
            hi=min(lo+64,N);raw[lo:hi]=forecast(model,h[lo:hi],f[lo:hi],expose)
            if lo==done or hi==N or lo//64%40==0:
                np.savez(dest/'partial.npz',raw=raw,done=hi);dump(dest/'progress.json',{'done':hi,'total':N,'seconds':round(time.time()-start,2)});print(arm,hi,'/',N,'seconds',round(time.time()-start,2),flush=True)
        np.savez_compressed(dest/'predictions.npz',raw=raw,prediction=np.maximum(raw,0));dump(dest/'done.json',dict(fingerprint,completed=True,seconds=round(time.time()-start,2)))
    base=forecast(model,h[:2],f[:2],True);changed=forecast(model,h[:2],f[:2]+10,True);hidden=forecast(model,h[:2],f[:2],False);hidden2=forecast(model,h[:2],f[:2]+10,False)
    masked=forecast(model,h[:2],f[:2],True,masked_fill=1e4);single=forecast(model,h[:1],f[:1],True)
    effects={'future_easy_changes_prediction':float(np.max(np.abs(base-changed))),'history_only_ignores_future_easy':float(np.max(np.abs(hidden-hidden2))),'masked_value_filler_does_not_affect_output':float(np.max(np.abs(base-masked))),'isolated_task_vs_batch_max_abs_difference':float(np.max(np.abs(base[:1]-single)))}
    assert effects['future_easy_changes_prediction']>0 and effects['history_only_ignores_future_easy']==0
    assert np.allclose(base,masked,rtol=1e-5,atol=1e-6) and np.allclose(base[:1],single,rtol=1e-4,atol=1e-5)
    dump(HERE/('试跑输入核验.json' if a.limit else '输入隔离核验.json'),dict(passed=True,no_future_chemistry_in_interface=True,**effects,fingerprint=fingerprint))
if __name__=='__main__':main()
