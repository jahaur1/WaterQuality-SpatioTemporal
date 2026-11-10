"""Local pretrained, frozen-weight +24h benchmark. No test-driven tuning."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['HF_MODULES_CACHE']=str(HERE/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,time,hashlib,argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
ROOT=HERE.parent;OLD=ROOT/'XLinear水质24小时条件预测_20261102/results'
SOURCE=ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv'
EASY=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];COLS=EASY+CHEM
CKPT=Path('D:/AI/TATO-main/CKPT')
MODELS={'Chronos-tiny':'Chronos-tiny','MOIRAI-small':'MOIRAI-small','MOIRAI-base':'MOIRAI-base','MOIRAI-large':'MOIRAI-large','Sundial-base':'sundial','Timer-base':'Timer-base-84m'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def save(df,p):df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')
def prepare():
    assert sha(SOURCE)=='9267d65df4c67f934339812b9fdea839696594c8c55af6b8f6fefb445d3cb7eb'
    d=pd.read_csv(SOURCE);d['监测时间']=pd.to_datetime(d['监测时间'],format='mixed')
    p=pd.read_csv(OLD/'02_精确24小时测试实测与预测.csv');p=p[p['分支']=='conditional_future_easy'].copy().reset_index(drop=True)
    for c in ['起报时间','目标时间','输入历史起点']:p[c]=pd.to_datetime(p[c],format='mixed')
    assert len(p)==23136 and p['站点ID'].nunique()==750
    groups={s:g.sort_values('监测时间').reset_index(drop=True) for s,g in d.groupby('站点ID')}
    h=np.full((len(p),32,9),np.nan,dtype=np.float32);y=np.empty((len(p),4),np.float32);f=np.empty((len(p),5),np.float32);lengths=[]
    for k,r in p.iterrows():
        g=groups[r['站点ID']];oi=int(r['起报站内索引']);ti=int(r['目标站内索引']);hist=g.iloc[oi-5:oi+1]
        assert len(hist)==6 and g.iloc[oi]['监测时间']==r['起报时间'] and g.iloc[ti]['监测时间']==r['目标时间']
        assert r['目标时间']-r['起报时间']==pd.Timedelta(hours=24)
        lag=((r['起报时间']-hist['监测时间'])/pd.Timedelta(hours=4)).to_numpy()
        assert np.all(lag==np.round(lag)) and lag.max()<=30
        n=int(lag.max())+1;grid=pd.date_range(hist['监测时间'].iloc[0],r['起报时间'],freq='4h')
        # Forward fill only within the six-observation history, never before its start.
        filled=hist.set_index('监测时间')[COLS].reindex(grid).ffill().to_numpy(np.float32)
        assert len(filled)==n and np.isfinite(filled).all()
        h[k,-n:]=filled;lengths.append(n);y[k]=g.iloc[ti][CHEM].to_numpy(float);f[k]=g.iloc[ti][EASY].to_numpy(float)
        assert np.allclose(y[k],r[[c+'_实测' for c in CHEM]].to_numpy(float),atol=2e-6,rtol=1e-6)
    np.savez_compressed(HERE/'inputs.npz',history=h,truth=y,future_easy=f,lengths=np.array(lengths))
    save(p[['站点ID','省份','流域','断面','起报时间','目标时间','输入历史起点','起报站内索引','目标站内索引']],HERE/'测试配对索引.csv')
    from safetensors import safe_open
    registry=[]
    for model,folder in MODELS.items():
        path=CKPT/folder/'model.safetensors'
        with safe_open(path,framework='pt',device='cpu') as weights:
            count=sum(int(np.prod(weights.get_slice(key).get_shape())) for key in weights.keys())
        registry.append({'模型':model,'参数量_权重张量元素数':count,'权重字节数':path.stat().st_size,'路径':str(path),'SHA256':sha(path)})
    save(pd.DataFrame(registry),HERE/'本地权重清单.csv')
    dump(HERE/'实验配置.json',{'source_sha256':sha(SOURCE),'test_index_sha256':sha(HERE/'测试配对索引.csv'),'old_predictions_sha256':sha(OLD/'02_精确24小时测试实测与预测.csv'),'stations':750,'test_pairs':len(p),'history_observations':6,'grid_hours':4,'prediction_steps':6,'context_grid':32,'max_history_hours':max(lengths)*4-4,'fill':'causal forward fill within last six observations; no target filling','samples':20,'point_forecast':'sample median (average of two central order statistics for even sample count)','seed':20261103,'tuning':'none; frozen pretrained weights; no fine-tuning; fixed patch8 for MOIRAI','MOIRAI_future':'only target-time 5 easy values observed; first 5 future grid steps masked; no future chemicals','Timer_padding':'normalize each window over available grid history; left pad to96 with normalized zero (window mean), explicit model length adaptation','Sundial_padding':'normalize over available history; left pad to32 with normalized zero; revin=False to avoid including padding in statistics','unknown_padding':'Chronos NaN; MOIRAI zero plus observed mask and past_is_pad'})
    print('PREPARED',len(p),flush=True)
def infer(model,arm,batch,limit):
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261103);np.random.seed(20261103)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this full comparison')
    z=np.load(HERE/'inputs.npz');h=z['history'];f=z['future_easy'];N=len(h) if not limit else min(limit,len(h));h=h[:N];f=f[:N]
    path=CKPT/MODELS[model];tag=model+'__'+arm;dest=HERE/('pilot' if limit else 'results')/tag;dest.mkdir(parents=True,exist_ok=True)
    config={'model':model,'arm':arm,'batch':batch,'samples':20,'limit':limit,'seed':20261103,'inputs_sha256':sha(HERE/'inputs.npz'),'weights_sha256':sha(path/'model.safetensors'),'torch':torch.__version__,'python':sys.version}
    if (dest/'done.json').exists():
        old=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert all(old[k]==v for k,v in config.items());print('ALREADY_DONE',tag);return
    from transformers import __version__ as tv
    config['transformers']=tv;start=time.time()
    if model.startswith('Chronos'):
        from chronos import ChronosPipeline
        pipeline=ChronosPipeline.from_pretrained(str(path),device_map='cuda',torch_dtype=torch.float32,local_files_only=True)
        pipeline.model.eval()
    elif model.startswith('MOIRAI'):
        from uni2ts.model.moirai import MoiraiModule,MoiraiForecast
        module=MoiraiModule.from_pretrained(str(path),local_files_only=True)
        pipeline=MoiraiForecast(module=module,prediction_length=6,target_dim=4,feat_dynamic_real_dim=5,past_feat_dynamic_real_dim=0,context_length=32,patch_size=8,num_samples=20).to('cuda').eval()
    else:
        from transformers import AutoModelForCausalLM
        pipeline=AutoModelForCausalLM.from_pretrained(str(path),trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to('cuda').eval()
    raw=np.full((N,4),np.nan,dtype=np.float32);done=0
    # Resume partial batches with the corresponding RNG state, for stochastic comparability.
    state=dest/'partial.npz';rngfile=dest/'rng.pt'
    if state.exists():
        old=np.load(state);raw[:]=old['raw'];done=int(old['done']);rng=torch.load(rngfile,weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    for lo in range(done,N,batch):
        hi=min(lo+batch,N);hh=h[lo:hi];valid=np.isfinite(hh);bs=hi-lo
        with torch.inference_mode():
            if model.startswith('Chronos'):
                x=torch.from_numpy(hh[:,:,5:].transpose(0,2,1).reshape(-1,32).copy())
                pred=pipeline.predict(x,prediction_length=6,num_samples=20)
                out=torch.quantile(pred[:,:,-1].float(),.5,dim=1).reshape(bs,4).cpu().numpy()
            elif model.startswith('MOIRAI'):
                dyn=np.concatenate([np.nan_to_num(hh[:,:,:5]),np.zeros((bs,6,5),np.float32)],axis=1)
                dm=np.concatenate([valid[:,:,:5],np.zeros((bs,6,5),bool)],axis=1)
                if arm=='future_easy':dyn[:,-1]=f[lo:hi];dm[:,-1]=True
                kwargs={'past_target':torch.tensor(np.nan_to_num(hh[:,:,5:]),device='cuda'),'past_observed_target':torch.tensor(valid[:,:,5:],device='cuda'),'past_is_pad':torch.tensor(~valid.any(-1),device='cuda'),'feat_dynamic_real':torch.tensor(dyn,dtype=torch.float32,device='cuda'),'observed_feat_dynamic_real':torch.tensor(dm,device='cuda')}
                samples=pipeline(**kwargs)
                assert samples.shape==(bs,20,6,4),samples.shape
                out=torch.quantile(samples[:,:,-1,:].float(),.5,dim=1).cpu().numpy()
            else:
                x=hh[:,:,5:].transpose(0,2,1).reshape(-1,32).copy();mu=np.nanmean(x,1,keepdims=True);sd=np.nanstd(x,1,keepdims=True);sd=np.where(sd>1e-5,sd,1.)
                norm=np.nan_to_num((x-mu)/sd)
                if model.startswith('Timer'):norm=np.pad(norm,((0,0),(64,0)))
                tx=torch.tensor(norm,device='cuda',dtype=torch.float32)
                if model.startswith('Timer'):
                    # One native output patch is already at least 6 steps; direct forward equals first generation step.
                    pp=pipeline(input_ids=tx,max_output_length=6,return_dict=True).logits
                    assert pp.shape==(bs*4,6),pp.shape
                    out=(pp[:,5:6].cpu().numpy()*sd+mu).reshape(bs,4)
                else:
                    pp=pipeline(input_ids=tx,max_output_length=6,num_samples=20,return_dict=True,revin=False).logits
                    assert pp.shape==(bs*4,20,6),pp.shape
                    out=(torch.quantile(pp[:,:,5].float(),.5,dim=1).cpu().numpy()[:,None]*sd+mu).reshape(bs,4)
        assert np.isfinite(out).all(),(tag,lo)
        raw[lo:hi]=out
        if lo==done or hi==N or (lo//batch)%20==0:
            np.savez(state,raw=raw,done=hi);torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},rngfile)
            dump(dest/'progress.json',{'done':hi,'total':N,'seconds_this_run':round(time.time()-start,2)})
            print(tag,hi,'/',N,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(dest/'predictions.npz',raw=raw,prediction=np.maximum(raw,0))
    config.update(seconds=round(time.time()-start,2),N=N,negative_predictions=int((raw<0).sum()),completed=True)
    dump(dest/'done.json',config);print('COMPLETED',tag,config['seconds'],flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','infer']);ap.add_argument('--model',choices=MODELS);ap.add_argument('--arm',default='history_only',choices=['history_only','future_easy']);ap.add_argument('--batch',type=int,default=128);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.action=='prepare':prepare()
    else:infer(a.model,a.arm,a.batch,a.limit)
if __name__=='__main__':main()
