"""Official frozen Timestral-209M, native NaN covariates, same +24h observed pairs."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,subprocess
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'source/src'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='2'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from timestral import TimestralForecaster
REF=HERE.parent/'预训练模型水质24小时评估_20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def forecast(pipe,h,f,expose,masked_fill=None):
    b=len(h);target=h[:,:,5:].transpose(0,2,1).copy();observed=np.isfinite(target)
    if masked_fill is not None:target=np.where(observed,target,masked_fill)
    known=np.full((b,5,38),np.nan,np.float32);known[:,:,:32]=h[:,:,:5].transpose(0,2,1)
    if expose:known[:,:,-1]=f
    assert np.isnan(known[:,:,32:37]).all()
    assert np.isfinite(known[:,:,-1]).all() if expose else np.isnan(known[:,:,32:]).all()
    # Context consists of four past chemical series only; no future target values are passed.
    assert target.shape==(b,4,32) and known.shape==(b,5,38)
    out=pipe.forecast(target,prediction_length=6,observation_mask=observed,known_covariates=known,symmetric_averaging=True,clip_nonnegative=False)
    assert out.quantiles.shape==(b,4,6,21),out.quantiles.shape
    result=out.median[:,:,5].float().cpu().numpy();assert np.isfinite(result).all();return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(20261104)
    z=np.load(REF/'inputs.npz');h=z['history'];f=z['future_easy'];N=min(len(h),a.limit) if a.limit else len(h)
    audit=json.loads((HERE/'下载SHA256核验.json').read_text(encoding='utf-8'));assert audit['passed'] and sha(HERE/'weights/model.safetensors')==audit['sha256']
    pipe=TimestralForecaster.from_pretrained(HERE/'weights',device='cuda',max_context=32,local_files_only=True)
    assert pipe.supports_covariates and pipe.supports_known_covariates
    for p in pipe.model.parameters():p.requires_grad_(False)
    git_sha=subprocess.check_output(['git','-c','safe.directory='+str((HERE/'source').as_posix()),'rev-parse','HEAD'],cwd=HERE/'source',text=True).strip()
    from safetensors import safe_open
    with safe_open(HERE/'weights/model.safetensors',framework='pt') as w:count=sum(int(np.prod(w.get_slice(k).get_shape())) for k in w.keys())
    fingerprint={'weights_sha256':audit['sha256'],'inputs_sha256':sha(REF/'inputs.npz'),'limit':a.limit,'N':N,'batch_windows':64,'precision':'float32','point_quantile':.5,'grid_hours':4,'prediction_length':6,'context_length':32,'fine_tune':False,'official_source_commit':git_sha,'parameter_tensor_elements':count,'native_api':'TimestralForecaster.forecast with observation_mask and NaN known_covariates','history_covariate_role':'same known_covariates in both arms; all future NaN in history_only','high_level_wrapper_interpolation_used':False,'symmetric_averaging':True,'quantile_real_cap_k':pipe.quantile_real_cap_k,'torch':torch.__version__}
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
            hi=min(lo+64,N);raw[lo:hi]=forecast(pipe,h[lo:hi],f[lo:hi],expose)
            if lo==done or hi==N or lo//64%40==0:
                np.savez(dest/'partial.npz',raw=raw,done=hi);dump(dest/'progress.json',{'done':hi,'total':N,'seconds':round(time.time()-start,2)});print(arm,hi,'/',N,'seconds',round(time.time()-start,2),flush=True)
        np.savez_compressed(dest/'predictions.npz',raw=raw,prediction=np.maximum(raw,0));dump(dest/'done.json',dict(fingerprint,completed=True,seconds=round(time.time()-start,2)))
    base=forecast(pipe,h[:2],f[:2],True);changed=forecast(pipe,h[:2],f[:2]+10,True);hidden=forecast(pipe,h[:2],f[:2],False);hidden2=forecast(pipe,h[:2],f[:2]+10,False)
    masked=forecast(pipe,h[:2],f[:2],True,masked_fill=1e4);single=forecast(pipe,h[:1],f[:1],True)
    effects={'future_easy_changes_prediction':float(np.max(np.abs(base-changed))),'history_only_ignores_future_easy':float(np.max(np.abs(hidden-hidden2))),'masked_target_filler_does_not_affect_output':float(np.max(np.abs(base-masked))),'isolated_task_vs_batch_max_abs_difference':float(np.max(np.abs(base[:1]-single)))}
    assert effects['future_easy_changes_prediction']>0 and effects['history_only_ignores_future_easy']==0
    assert np.allclose(base,masked,rtol=1e-5,atol=1e-6) and np.allclose(base[:1],single,rtol=1e-4,atol=1e-5)
    dump(HERE/('试跑输入核验.json' if a.limit else '输入隔离核验.json'),dict(passed=True,no_future_chemistry_in_interface=True,**effects,fingerprint=fingerprint))
if __name__=='__main__':main()
