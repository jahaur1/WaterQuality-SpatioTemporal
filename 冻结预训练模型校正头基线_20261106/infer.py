"""Frozen conditional backbones; reuse exact old test predictions, infer pre-test windows."""
from pathlib import Path
import sys,os,argparse,importlib.util,json,time,hashlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Timestral','Chronos2'],required=True);a=ap.parse_args()
    folder=ROOT/('Timestral209M水质24小时评估_20261104' if a.model=='Timestral' else 'Chronos2水质24小时评估_20261103')
    spec=importlib.util.spec_from_file_location('frozen_runner',folder/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.set_num_threads(2);torch.manual_seed(20261106)
    if a.model=='Timestral':
        pipe=mod.TimestralForecaster.from_pretrained(folder/'weights',device='cuda',max_context=32,local_files_only=True);batch=64
    else:
        pipe=mod.Chronos2Pipeline.from_pretrained(str(folder/'weights'),device_map='cuda',torch_dtype=torch.float32,local_files_only=True);batch=128
    pipe.model.eval()
    for p in pipe.model.parameters():p.requires_grad_(False)
    assert not any(p.requires_grad for p in pipe.model.parameters())
    z=np.load(HERE/'inputs.npz');h=z['history'];f=z['future_easy'];n=len(h)
    dest=HERE/a.model;dest.mkdir(exist_ok=True)
    fingerprint={'inputs_sha256':sha(HERE/'inputs.npz'),'weights_sha256':sha(folder/'weights/model.safetensors'),'runner_sha256':sha(folder/'run.py'),'all_backbone_parameters_frozen':True,'backbone_training':False,'same_previous_conditional_forecast_function':True}
    pred=np.full((n,4),np.nan,np.float32)
    old=np.load(folder/'results/future_easy/predictions.npz')['raw'];pred[z['test_new_indices']]=old[z['test_old_indices']]
    cache=dest/'predictions.npz'
    if cache.exists():
        audit=json.loads((dest/'fingerprint.json').read_text());assert audit==fingerprint;pred=np.load(cache)['raw']
    else:(dest/'fingerprint.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    missing=np.flatnonzero(~np.isfinite(pred).all(1));start=time.time()
    with torch.inference_mode():
        for lo in range(0,len(missing),batch):
            ix=missing[lo:lo+batch];pred[ix]=mod.forecast(pipe,h[ix],f[ix],True)
            if lo==0 or (lo//batch)%20==0 or lo+batch>=len(missing):
                np.savez_compressed(cache,raw=pred)
                print(a.model,min(lo+batch,len(missing)),'/',len(missing),'new windows; seconds',round(time.time()-start,1),flush=True)
    assert np.isfinite(pred).all()
    ii=z['test_new_indices'];jj=z['test_old_indices'];assert np.array_equal(pred[ii],old[jj])
    # Repeat a few new windows separately to verify native task isolation and saved inference.
    ix=missing[:2]
    if len(ix):
        with torch.inference_mode():replay=mod.forecast(pipe,h[ix],f[ix],True)
        assert np.allclose(pred[ix],replay,rtol=1e-4,atol=1e-5)
    (dest/'完成核验.json').write_text(json.dumps(dict(fingerprint,passed=True,total_windows=n,reused_test_windows=len(ii),new_windows=len(missing),seconds=time.time()-start),indent=2),encoding='utf-8')
if __name__=='__main__':main()
