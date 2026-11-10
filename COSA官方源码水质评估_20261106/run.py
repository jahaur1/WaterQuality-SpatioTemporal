"""Native COSA core on cached frozen predictions with timestamp-safe delayed labels."""
from pathlib import Path
import sys,json,time,hashlib,subprocess
from collections import deque
import numpy as np
import pandas as pd
import torch
from native import CausalEndpointAdapter,SOURCE,OPT
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;CACHE=ROOT/'冻结预训练模型校正头基线_20261106'
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(df,name):df.to_csv(HERE/name,index=False,encoding='utf-8-sig',float_format='%.12g')
def score(y,p):
    e=p-y;sst=np.sum((y-y.mean())**2);tol=1e-12*max(1.,np.dot(y,y))
    return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),R2=float(1-np.dot(e,e)/sst) if len(y)>=2 and sst>tol else np.nan)
def main():
    torch.set_num_threads(1);idx=pd.read_csv(CACHE/'配对索引.csv');idx=idx[idx.test].reset_index(drop=True)
    for c in ['起报时间','目标时间']:idx[c]=pd.to_datetime(idx[c],format='mixed')
    z=np.load(CACHE/'inputs.npz');te=pd.read_csv(CACHE/'配对索引.csv').test.to_numpy(bool)
    truth=z['truth'][te];stats=json.loads((CACHE/'站点训练标准化.json').read_text(encoding='utf-8'))
    commit=subprocess.check_output(['git','-c','safe.directory='+str((HERE/'source').as_posix()),'rev-parse','HEAD'],cwd=HERE/'source',text=True).strip()
    config={'official_repository':'https://github.com/bigbases/COSA_ICLR2026','official_commit':commit,'source_COSA_sha256':sha(SOURCE),'source_optimizer_sha256':sha(OPT),'native_wrapper_sha256':sha(HERE/'native.py'),'runner_sha256':sha(__file__),'context_size':10,'steps_per_completed_pair':3,'lr':.001,'l2_reg':.0001,'optimizer_weight_decay':.0001,'adaptive_lr':True,'fast_adaptation':True,'var_wise_gating':True,'adapter_type':'Linear','parameter_count':52,'pred_len':1,'endpoint_hours':24,'station_scope':'separate fresh adapter per station and backbone','initialization':'author Xavier gain .1, zero bias, zero tanh gates; fixed seed20261106+numeric station ID','partial_adapt':False,'adjust_already_emitted_predictions':False,'paas':False,'batch_size':1,'label_release':'past prediction target timestamp <= current origin timestamp','warm_start':False,'training_before_test':False,'online_test_updates':True,'normalization':'fixed per-station raw observations before Apr29; same cache as preceding experiment','task_adaptations':['Official class/function ASTs execute unchanged; bypass framework init and data loader only','Four chemistry channels and only +24h endpoint pred_len1','Batch1 per station, timestamps replace row-based completion schedule','Partial horizon updates, retrospective prediction adjustment, PAAS disabled for irregular endpoint task','Context tensor allocation moved from hardcoded CUDA to CPU; values/order/padding match native method','Use byte-identical previously frozen conditional predictions; no backbone or future chemical input updates']}
    (HERE/'实验配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    results={};updates=[];stationrecords=[];start=time.time()
    for backbone in ['Timestral','Chronos2']:
        previous=np.load(CACHE/backbone/'predictions.npz')['raw'][te]
        results[backbone+'_冻结原模型']=previous.astype(float);output=np.full(previous.shape,np.nan,float)
        dest=HERE/backbone;dest.mkdir(exist_ok=True);(dest/'stations').mkdir(exist_ok=True)
        for number,(sid,ix) in enumerate(idx.groupby('站点ID',sort=True).indices.items()):
            ix=np.asarray(ix);q=idx.iloc[ix];assert q['起报时间'].is_monotonic_increasing
            sc=stats[sid]['final'];mu=np.array(sc['mean'][5:]);sd=np.array(sc['scale'][5:]);seed=20261106+int(sid[1:]);torch.manual_seed(seed)
            state=CausalEndpointAdapter();assert sum(p.numel() for p in state.output_adapter.parameters())==52
            predz=torch.tensor(((previous[ix]-mu)/sd)[:,None,:],dtype=torch.float32)
            pending=deque();nc=0;origins=q['起报时间'].tolist();targets=q['目标时间'].tolist();emissions=[]
            for j,origin in enumerate(origins):
                # Reveal chemistry labels only after their actual timestamps arrive.
                while pending and targets[pending[0][0]]<=origin:
                    k,ctx=pending.popleft();target=targets[k];assert target<=origin
                    obs=torch.tensor(((truth[ix[k]]-mu)/sd)[None,None,:],dtype=torch.float32)
                    state.observe_completed(predz[k:k+1],obs,ctx,k,k)
                    updates.append({'模型':backbone,'站点ID':sid,'原预测起报时间':origins[k],'标签目标时间':target,'实际更新时间':origin,'梯度步数':3,'上下文均值来自此前已观测标签':True});nc+=1
                context=state.available_context();normalized=state.predict(predz[j:j+1],context)[0,0].numpy()
                output[ix[j]]=normalized*sd+mu
                if j==0:assert np.allclose(normalized,predz[j,0].numpy(),rtol=0,atol=0)
                emissions.append({'origin':str(origin),'target':str(targets[j]),'completed_pairs_before_prediction':nc})
                pending.append((j,context.clone()))
            torch.save({'state_dict':state.output_adapter.state_dict(),'station':sid,'seed':seed,'completed_updates':nc,'gradient_steps':state.n_adapt,'scaler':sc,'emissions':emissions},dest/'stations'/f'{sid}.pt')
            stationrecords.append({'模型':backbone,'站点ID':sid,'测试配对数':len(ix),'成熟标签更新次数':nc,'梯度步数':state.n_adapt,'未成熟尾部配对数':len(pending),'最终门控最大绝对值':float(torch.tanh(state.output_adapter.gate).abs().max().detach())})
            if number%100==0 or number==749:print(backbone,number+1,'/750; seconds',round(time.time()-start,1),flush=True)
        assert np.isfinite(output).all();results[backbone+'_COSA官方核心_因果在线']=output
    results['持久性']=z['last_chemical'][te]
    np.savez_compressed(HERE/'test_predictions.npz',truth=truth,**results)
    save(pd.DataFrame(updates),'标签释放与更新记录.csv');save(pd.DataFrame(stationrecords),'逐站点在线适配记录.csv');save(idx,'测试配对索引.csv')
    rows=[]
    for name,raw in results.items():
        pred=np.maximum(raw,0)
        for sid,ix in idx.groupby('站点ID').indices.items():
            meta=idx.iloc[ix[0]]
            for j,target in enumerate(CHEM):rows.append({'模型':name,'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面'],'目标变量':target,'测试配对数':len(ix),**score(truth[ix,j],pred[ix,j])})
    M=pd.DataFrame(rows);save(M,'01_逐站点模型误差.csv');summary=[];counts=[]
    for (name,target),g in M.groupby(['模型','目标变量'],sort=False):summary.append({'模型':name,'目标变量':target,'R2有效站点数':int(g.R2.notna().sum()),'R2均值':g.R2.mean(),'R2中位数':g.R2.median(),'MAE中位数':g.MAE.median(),'RMSE中位数':g.RMSE.median(),'R2大于0.5站点数':int(g.R2.gt(.5).sum()),'R2大于0.8站点数':int(g.R2.gt(.8).sum())})
    A=pd.DataFrame(summary);save(A,'02_逐指标汇总.csv')
    for name,g in M.groupby('模型',sort=False):counts.append({'模型':name,'四指标等权平均R2':A[A['模型'].eq(name)]['R2均值'].mean(),'任一项R2大于0.5站点数':g.loc[g.R2.gt(.5),'站点ID'].nunique(),'任一项R2大于0.8站点数':g.loc[g.R2.gt(.8),'站点ID'].nunique()})
    C=pd.DataFrame(counts);save(C,'03_总体比较.csv');gain=[]
    for backbone in ['Timestral','Chronos2']:
        b=M[M['模型'].eq(backbone+'_冻结原模型')];c=M[M['模型'].eq(backbone+'_COSA官方核心_因果在线')];v=c.merge(b,on=['站点ID','目标变量'],suffixes=('_COSA','_原始'),validate='one_to_one');v['R2提高量']=v.R2_COSA-v.R2_原始;v['MAE降低量']=v.MAE_原始-v.MAE_COSA;gain.append(v)
    save(pd.concat(gain),'04_逐站点校正收益.csv')
    (HERE/'完成记录.json').write_text(json.dumps({'passed_runner_assertions':True,'stations':750,'test_pairs':len(idx),'metrics':len(M),'online_updates':len(updates),'seconds':time.time()-start},indent=2),encoding='utf-8')
    print(C.to_string(index=False));print(A.to_string(index=False))
if __name__=='__main__':main()
