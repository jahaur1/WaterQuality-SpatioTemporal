"""Exact +24-hour observed-pair forecasting, two prespecified XLinear arms.

History: six observations through the origin, with 5 easy + 4 chemical inputs.
Conditional arm additionally receives the target-time five easy measurements.
The ablation receives zeros instead of those future easy covariates.
No interpolation, cross-station windows, or future chemical inputs are allowed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,hashlib,json,random,sys,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'30天_750站点数据集_20250410-20250510'
SOURCE=BASE/'水质数据_750站点_20250410-20250510.csv'
EXPECTED_SOURCE_SHA='9267d65df4c67f934339812b9fdea839696594c8c55af6b8f6fefb445d3cb7eb'
EASY=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
ALL=EASY+CHEM
SELECT_CUT=pd.Timestamp('2025-04-24')
VAL_START=pd.Timestamp('2025-04-25');VAL_END=pd.Timestamp('2025-04-29')
REFIT_CUT=pd.Timestamp('2025-04-29')
TEST_START=pd.Timestamp('2025-04-30');TEST_END=pd.Timestamp('2025-05-10')
ARMS=[('conditional_future_easy',True),('history_only',False)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
def save(frame,p):frame.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')
def seed(v):
    random.seed(v);np.random.seed(v);torch.manual_seed(v)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(v)
def model_for(conf,use_future):
    from adapter import XLinearConditional24
    return XLinearConditional24(seq_len=conf['seq_len'],d_model=conf['d_model'],t_ff=conf['t_ff'],c_ff=conf['c_ff'],dropout=conf['dropout'],use_future=use_future)
def score(y,p):
    if not len(y) or not np.isfinite(p).all():return {'MAE':np.nan,'RMSE':np.nan,'偏差':np.nan,'R2':np.nan}
    e=p-y;sst=np.sum((y-y.mean())**2);tol=1e-12*max(1.,float(np.sum(y*y)))
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'偏差':float(e.mean()),'R2':float(1-np.sum(e*e)/sst) if len(y)>=2 and sst>tol else np.nan}
def record_scaler(s):return {'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'n_samples':int(s.n_samples_seen_)}

def pairs_for(g,L=6):
    g=g.sort_values('监测时间').reset_index(drop=True)
    times=g['监测时间'];assert g['站点ID'].nunique()==1 and (times.diff().dropna()>pd.Timedelta(0)).all()
    lookup={t:i for i,t in enumerate(times)};pairs=[];cold=0;all_exact=0
    for origin,t in enumerate(times):
        target=lookup.get(t+pd.Timedelta(hours=24))
        if target is None:continue
        all_exact+=1
        if origin<L-1:cold+=1;continue
        pairs.append((origin,target))
    origins=np.array([i for i,j in pairs],dtype=int);targets=np.array([j for i,j in pairs],dtype=int)
    history=np.stack([np.arange(i-L+1,i+1) for i in origins]) if len(origins) else np.empty((0,L),dtype=int)
    ot=times.iloc[origins].reset_index(drop=True);tt=times.iloc[targets].reset_index(drop=True)
    assert ((tt-ot)==pd.Timedelta(hours=24)).all()
    if len(origins):assert (history[:,-1]==origins).all() and (history.max(1)==origins).all() and (targets>origins).all()
    masks={
        'selection_train':(tt<SELECT_CUT).to_numpy(),
        'validation':((tt>=VAL_START)&(tt<VAL_END)&(ot>=SELECT_CUT)).to_numpy(),
        'refit_train':(tt<REFIT_CUT).to_numpy(),
        'test':((tt>=TEST_START)&(tt<=TEST_END)&(ot>=REFIT_CUT)).to_numpy(),
    }
    reasons=[]
    for part,minn,label in [('selection_train',3,'选epoch训练配对不足3'),('validation',1,'验证配对不足1'),('test',1,'测试配对不足1')]:
        if masks[part].sum()<minn:reasons.append(label)
    if masks['validation'].any():assert tt[masks['validation']].max()<REFIT_CUT and ot[masks['validation']].min()>=SELECT_CUT
    if masks['test'].any():
        minimum_origin=ot[masks['test']].min()
        if masks['refit_train'].any():assert tt[masks['refit_train']].max()<minimum_origin
        if masks['validation'].any():assert tt[masks['validation']].max()<minimum_origin
    meta=g.iloc[0]
    coverage={'站点ID':meta['站点ID'],'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面名称'],'原始观测数':len(g),'全部精确24小时配对数':all_exact,'历史不足6次丢弃配对数':cold,'有效6次历史精确配对数':len(pairs),**{f'{k}_配对数':int(v.sum()) for k,v in masks.items()},'可评估':int(not reasons),'不可评估原因':'；'.join(reasons),'原始训练Scaler记录数_选择':int((times<SELECT_CUT).sum()),'原始训练Scaler记录数_重拟合':int((times<REFIT_CUT).sum()),'配对最大历史跨度小时':float(((ot-times.iloc[history[:,0]].reset_index(drop=True))/pd.Timedelta(hours=1)).max()) if len(pairs) else np.nan}
    return g,origins,targets,history,ot,tt,masks,coverage

def build_tensors(raw,origins,targets,history,masks,scaler,device):
    z=scaler.transform(raw)
    # Future branch reads only the first five channels. Chemical values at targets are labels only.
    h=z[history];f=((raw[targets,:5]-scaler.mean_[:5])/scaler.scale_[:5])[:,None,:];y=(raw[targets,5:]-scaler.mean_[5:])/scaler.scale_[5:]
    result={}
    for name,mask in masks.items():
        result[name]=(torch.tensor(h[mask],dtype=torch.float32,device=device),torch.tensor(f[mask],dtype=torch.float32,device=device),torch.tensor(y[mask],dtype=torch.float32,device=device))
    return result

def evaluate_arm(g,origins,targets,history,ot,tt,masks,coverage,conf,device,dest,arm,use_future,fingerprint):
    raw=g[ALL].to_numpy(float);times=g['监测时间'];sid=g.iloc[0]['站点ID'];seedvalue=2025+int(sid[1:])
    testmask=masks['test'];oi=origins[testmask];ti=targets[testmask];truth=raw[ti,5:]
    persist=raw[oi,5:];meantrain=np.tile(raw[(times<REFIT_CUT).to_numpy(),5:].mean(0),(len(ti),1))
    predictions=np.full((len(ti),4),np.nan);raw_predictions=predictions.copy();bestepoch=0;best=np.nan;epochsran=0;trace=[];start=time.time()
    selection_scaler=None;final_scaler=None
    if coverage['可评估']:
        selection_scaler=StandardScaler().fit(raw[(times<SELECT_CUT).to_numpy()])
        tensors=build_tensors(raw,origins,targets,history,masks,selection_scaler,device)
        htrain,ftrain,ytrain=tensors['selection_train'];hval,fval,yval=tensors['validation']
        # Explicitly suppress future inputs in the ablation, even before adapter routing.
        if not use_future:ftrain=torch.zeros_like(ftrain);fval=torch.zeros_like(fval)
        seed(seedvalue);model=model_for(conf,use_future).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=conf['lr']);criterion=torch.nn.MSELoss()
        best=float('inf');bad=0
        for epoch in range(1,conf['max_epochs']+1):
            model.train();optimizer.zero_grad(set_to_none=True);loss=criterion(model(htrain,ftrain),ytrain);assert torch.isfinite(loss);loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():vloss=float(criterion(model(hval,fval),yval).item())
            assert np.isfinite(vloss)
            if vloss<best:best=vloss;bestepoch=epoch;bad=0
            else:bad+=1
            trace.append({'阶段':'选择epoch','epoch':epoch,'训练标准化MSE':float(loss.item()),'验证标准化MSE':vloss})
            if bad>=conf['patience']:break
        epochsran=len(trace);assert bestepoch>0
        del tensors,model,optimizer,htrain,ftrain,ytrain,hval,fval,yval
        final_scaler=StandardScaler().fit(raw[(times<REFIT_CUT).to_numpy()])
        tensors=build_tensors(raw,origins,targets,history,masks,final_scaler,device)
        hr,fr,yr=tensors['refit_train'];ht,ft,yt=tensors['test']
        if not use_future:fr=torch.zeros_like(fr);ft=torch.zeros_like(ft)
        seed(seedvalue);model=model_for(conf,use_future).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=conf['lr']);criterion=torch.nn.MSELoss()
        for epoch in range(1,bestepoch+1):
            model.train();optimizer.zero_grad(set_to_none=True);loss=criterion(model(hr,fr),yr);assert torch.isfinite(loss);loss.backward();optimizer.step()
            trace.append({'阶段':'起报前重新拟合','epoch':epoch,'训练标准化MSE':float(loss.item()),'验证标准化MSE':np.nan})
        model.eval()
        with torch.no_grad():pz=model(ht,ft).cpu().numpy()
        raw_predictions=pz*final_scaler.scale_[5:]+final_scaler.mean_[5:];predictions=np.maximum(raw_predictions,0)
        assert np.isfinite(raw_predictions).all()
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'station':sid,'arm':arm,'use_future':use_future,'config':conf,'seed':seedvalue,'best_epoch':bestepoch,'scaler':record_scaler(final_scaler),'history_columns':ALL,'future_columns':EASY,'target_columns':CHEM},dest/'model.pt')
        del tensors,model,optimizer,hr,fr,yr,ht,ft,yt
    meta=g.iloc[0];metricrows=[]
    for j,target in enumerate(CHEM):
        sc=score(truth[:,j],predictions[:,j]);pb=score(truth[:,j],persist[:,j]);mb=score(truth[:,j],meantrain[:,j])
        metricrows.append({'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面名称'],'分支':arm,'目标变量':target,'可评估':coverage['可评估'],'不可评估原因':coverage['不可评估原因'],'精确24小时测试配对数':len(ti),'模型有效测试数':len(ti) if coverage['可评估'] else 0,'测试不足20对':int(len(ti)<20),'最优epoch':bestepoch,'选择实际epoch数':epochsran,'验证最小标准化MSE':best,**sc,'未截断MAE':score(truth[:,j],raw_predictions[:,j])['MAE'],'原始负预测次数':int((raw_predictions[:,j]<0).sum()) if coverage['可评估'] else 0,**{f'起报时化学值持久性_{k}':v for k,v in pb.items()},**{f'训练均值基线_{k}':v for k,v in mb.items()},'较持久性MAE改善比例':1-sc['MAE']/pb['MAE'] if pb['MAE']>0 else np.nan,'测试实测标准差':float(truth[:,j].std()) if len(ti) else np.nan,'R2可计算':int(np.isfinite(sc['R2']))})
    frame=pd.DataFrame({'站点ID':[sid]*len(ti),'省份':[meta['省份']]*len(ti),'流域':[meta['流域']]*len(ti),'断面':[meta['断面名称']]*len(ti),'分支':[arm]*len(ti),'可评估':[coverage['可评估']]*len(ti),'不可评估原因':[coverage['不可评估原因']]*len(ti),'起报时间':ot[testmask].to_numpy(),'目标时间':tt[testmask].to_numpy(),'输入历史起点':times.iloc[history[testmask,0]].to_numpy(),'历史跨度小时':((ot[testmask].reset_index(drop=True)-times.iloc[history[testmask,0]].reset_index(drop=True))/pd.Timedelta(hours=1)).to_numpy(),'起报站内索引':oi,'目标站内索引':ti})
    for j,col in enumerate(EASY):frame[col+'_目标时刻易测值']=raw[ti,j]
    for j,target in enumerate(CHEM):
        for label,values in [('实测',truth),('预测',predictions),('未截断预测',raw_predictions),('起报时化学值持久性',persist),('训练均值基线',meantrain)]:frame[target+'_'+label]=values[:,j]
    save(pd.DataFrame(metricrows),dest/'metrics.csv');save(frame,dest/'predictions.csv')
    save(pd.DataFrame(trace,columns=['阶段','epoch','训练标准化MSE','验证标准化MSE']),dest/'history.csv')
    metadata={'station':sid,'arm':arm,'use_future':use_future,'fingerprint':fingerprint,'seed':seedvalue,'status':'evaluated' if coverage['可评估'] else 'not_evaluable','reason':coverage['不可评估原因'],'coverage':coverage,'best_epoch':bestepoch,'validation_best_mse':best,'epochs_selection':epochsran,'epochs_refit':bestepoch,'selection_scaler':record_scaler(selection_scaler) if selection_scaler else None,'final_scaler':record_scaler(final_scaler) if final_scaler else None,'selection_scaler_latest_time':str(times[times<SELECT_CUT].max()),'final_scaler_latest_time':str(times[times<REFIT_CUT].max()),'selection_latest_label':str(tt[masks['selection_train']].max()),'validation_latest_label':str(tt[masks['validation']].max()),'test_earliest_origin':str(ot[testmask].min()),'final_train_latest_label':str(tt[masks['refit_train']].max()),'seconds':round(time.time()-start,3)}
    dump(dest/'done.json',metadata)
    return pd.DataFrame(metricrows),frame,metadata

def aggregate(g,basin,arm,target):
    eligible=g[g['可评估']==1];r=eligible['R2'].dropna();better=eligible['MAE']<eligible['起报时化学值持久性_MAE']
    return {'流域':basin,'分支':arm,'目标变量':target,'基础站点数':len(g),'可评估站点数':len(eligible),'不可评估站点数':len(g)-len(eligible),'R2可计算站点数':len(r),'MAE中位数':float(eligible['MAE'].median()),'RMSE中位数':float(eligible['RMSE'].median()),'R2中位数':float(r.median()) if len(r) else np.nan,'R2_P25':float(r.quantile(.25)) if len(r) else np.nan,'R2_P75':float(r.quantile(.75)) if len(r) else np.nan,'持久性MAE中位数_同可评估站点':float(eligible['起报时化学值持久性_MAE'].median()),'训练均值MAE中位数_同可评估站点':float(eligible['训练均值基线_MAE'].median()),'MAE优于持久性站点数':int(better.sum()),'MAE优于持久性占可评估比例':float(better.mean()) if len(eligible) else np.nan,'说明':'比较比例不是推导成功率；真实未来易测值为条件信息'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--patience',type=int,default=10);parser.add_argument('--output',default='results');parser.add_argument('--cpu',action='store_true');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    assert args.epochs>=1 and args.patience>=1 and args.limit>=0
    out=(HERE/args.output).resolve();assert out.is_relative_to(HERE.resolve());out.mkdir(exist_ok=True);(out/'stations').mkdir(exist_ok=True)
    require_files=[SOURCE] if args.check_only else [SOURCE,HERE/'adapter.py',HERE/'vendor_XLinear.py']
    for p in require_files:assert p.exists(),p
    assert sha(SOURCE)==EXPECTED_SOURCE_SHA
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    conf={'seq_len':6,'d_model':16,'t_ff':16,'c_ff':16,'dropout':.1,'lr':.001,'max_epochs':args.epochs,'patience':args.patience,'horizon_hours':24,'history_columns':ALL,'future_easy_columns':EASY,'target_columns':CHEM,'selection_label_before':str(SELECT_CUT),'selection_scaler_raw_before':str(SELECT_CUT),'validation_label_start':str(VAL_START),'validation_label_before':str(VAL_END),'final_label_before':str(REFIT_CUT),'final_scaler_raw_before':str(REFIT_CUT),'test_label_start':str(TEST_START),'test_label_end_inclusive':str(TEST_END),'minimum_selection_train_pairs':3,'minimum_validation_pairs':1,'minimum_test_pairs':1,'seed_base':2025,'refit':True,'arms':[a for a,b in ARMS],'normalization':'one training-only 9-column scaler; chemical outputs use columns 5:9; future easy uses 0:5','loss':'four standardized target MSE averaged','negative_prediction_postprocess':'clip zero, raw output also saved'}
    provenance={'source_dataset_sha256':sha(SOURCE),'runner_sha256':sha(__file__),'adapter_sha256':sha(HERE/'adapter.py') if (HERE/'adapter.py').exists() else None,'vendor_sha256':sha(HERE/'vendor_XLinear.py') if (HERE/'vendor_XLinear.py').exists() else None}
    fingerprint=hashlib.sha256(json.dumps({'config':conf,**provenance},sort_keys=True).encode()).hexdigest()
    dump(out/'config.json',{'config':conf,'fingerprint':fingerprint,**provenance})
    dump(out/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None})
    df=pd.read_csv(SOURCE);df['监测时间']=pd.to_datetime(df['监测时间']);assert len(df)==79940 and df['站点ID'].nunique()==750 and not df.duplicated(['站点ID','监测时间']).any() and np.isfinite(df[ALL].to_numpy(float)).all()
    groups=list(df.groupby('站点ID'));groups=groups[:args.limit] if args.limit else groups
    bundles=[pairs_for(g,conf['seq_len']) for sid,g in groups];coverage=pd.DataFrame([b[-1] for b in bundles]);save(coverage,out/'03_站点配对覆盖与不可评估原因.csv')
    precheck={'基础站点数':len(groups),'可评估站点数':int(coverage['可评估'].sum()),'不可评估站点数':int((coverage['可评估']==0).sum()),'原始观测数':int(coverage['原始观测数'].sum()),'全部精确24小时配对数':int(coverage['全部精确24小时配对数'].sum()),'6次历史可用精确配对数':int(coverage['有效6次历史精确配对数'].sum()),'全部测试精确配对数':int(coverage['test_配对数'].sum()),'模型可评估站点测试精确配对数':int(coverage.loc[coverage['可评估']==1,'test_配对数'].sum()),'无插值':True,'严格24小时':True,'无跨站或未来化学输入':True,'验证标签全部早于最早最终测试起报':True,'check_only':args.check_only,'fingerprint':fingerprint}
    dump(out/'pairing_precheck.json',precheck);print(json.dumps(precheck,ensure_ascii=False,indent=2),flush=True)
    if args.check_only:return
    allm=[];allp=[];allmeta=[];started=time.time()
    for i,bundle in enumerate(bundles):
        g,origins,targets,history,ot,tt,masks,cov=bundle;sid=g.iloc[0]['站点ID']
        for arm,use_future in ARMS:
            dest=out/'stations'/sid/arm;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'done.json').exists():
                meta=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert meta['fingerprint']==fingerprint,('stale result/config; choose a fresh output folder',sid,arm)
                m=pd.read_csv(dest/'metrics.csv');p=pd.read_csv(dest/'predictions.csv')
            else:m,p,meta=evaluate_arm(*bundle[:-1],cov,conf,device,dest,arm,use_future,fingerprint)
            assert len(m)==4;allm.append(m);allp.append(p);allmeta.append(meta)
        if i%25==0 or i+1==len(bundles):
            elapsed=time.time()-started;print(f'Completed {i+1}/{len(groups)} stations x 2 arms; elapsed {elapsed:.1f}s',flush=True)
            dump(out/'progress.json',{'completed_stations':i+1,'total':len(groups),'elapsed_seconds':round(elapsed,2),'last_station':sid})
    M=pd.concat(allm,ignore_index=True);P=pd.concat(allp,ignore_index=True);assert len(M)==8*len(groups)
    assert not M.duplicated(['站点ID','分支','目标变量']).any()
    assert not P.duplicated(['站点ID','分支','起报时间','目标时间']).any()
    observed=P[P['可评估']==1]
    for target in CHEM:assert np.isfinite(observed[target+'_预测']).all() and (observed[target+'_预测']>=0).all()
    assert (M.loc[M['可评估']==1,'RMSE']+1e-10>=M.loc[M['可评估']==1,'MAE']).all()
    assert len(P)==2*precheck['全部测试精确配对数']
    save(M,out/'01_逐站点两分支四目标误差.csv');save(P,out/'02_精确24小时测试实测与预测.csv')
    A=pd.DataFrame([aggregate(g,'全部流域',arm,target) for (arm,target),g in M.groupby(['分支','目标变量'],sort=False)])
    B=pd.DataFrame([aggregate(g,basin,arm,target) for (basin,arm,target),g in M.groupby(['流域','分支','目标变量'])]);save(A,out/'04_总体两分支与基线对比.csv');save(B,out/'05_分流域两分支与基线对比.csv')
    scalar_meta=[{k:v for k,v in m.items() if not isinstance(v,(dict,list))} for m in allmeta];save(pd.DataFrame(scalar_meta),out/'06_逐站点分支训练与时间记录.csv')
    # Independently recompute all saved model and persistence errors without training imports.
    reread=pd.read_csv(out/'02_精确24小时测试实测与预测.csv');lookup=M.set_index(['站点ID','分支','目标变量'])
    for (sid,arm),g in reread.groupby(['站点ID','分支']):
        require_eligible=bool(g['可评估'].iloc[0])
        for target in CHEM:
            r=lookup.loc[(sid,arm,target)];sc=score(g[target+'_实测'].to_numpy(),g[target+'_预测'].to_numpy());pb=score(g[target+'_实测'].to_numpy(),g[target+'_起报时化学值持久性'].to_numpy())
            for name in ['MAE','RMSE','偏差','R2']:
                assert np.isclose(sc[name],r[name],rtol=1e-6,atol=1e-7,equal_nan=True),(sid,arm,target,name)
                assert np.isclose(pb[name],r['起报时化学值持久性_'+name],rtol=1e-6,atol=1e-7,equal_nan=True)
            if not require_eligible:assert g[target+'_预测'].isna().all()
    if not args.limit:assert len(M)==6000 and len(coverage)==750
    assert sha(SOURCE)==provenance['source_dataset_sha256']
    summary={**precheck,'站点目标分支指标数':len(M),'两分支测试记录合计':len(P),'总体结果':A.to_dict('records'),'两分支不按测试选择优者':True,'持久性含义':'每个起报时刻最后一条化学实测值，随起报时刻更新','条件信息含义':'目标时刻五项易测真值，条件预测不是无条件未来预测','原始化学历史输入':True,'预测目标':'恰好24小时后的四化学指标','验证标签最迟边界':str(VAL_END),'所有scaler仅各自起报前原始训练数据拟合':True,'独立读取保存预测复算误差通过':True,'不可评估站点保留':True,'推导合格率':'未设容许误差，不计算','elapsed_seconds':round(time.time()-started,2),'fingerprint':fingerprint}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
