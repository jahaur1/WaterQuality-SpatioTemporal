"""XLinear backbone adapted for five measured inputs -> four same-time chemical labels."""
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
from adapter import XLinearSoftSensor

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SOURCE=ROOT/'30天_750站点数据集_20250410-20250510'/'水质数据_750站点_20250410-20250510.csv'
RIDGE=SOURCE.parent/'测5项估4项_时间留出预测基线'/'01_750站点各化学指标预测误差.csv'
XCOL=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']
YCOL=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
VAL=pd.Timestamp('2025-04-25');TEST=pd.Timestamp('2025-04-30');END=pd.Timestamp('2025-05-10')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
def save(frame,p):frame.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')
def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
def score(y,p):
    e=p-y;denom=np.sum((y-y.mean())**2);tol=1e-12*max(1.,float(np.sum(y*y)))
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.square(e).mean())),'偏差':float(e.mean()),'R2':float(1-np.sum(e*e)/denom) if denom>tol else np.nan}
def scaler_record(s):return {'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'n_samples':int(s.n_samples_seen_)}
def new_model(conf):return XLinearSoftSensor(seq_len=conf['seq_len'],d_model=conf['d_model'],t_ff=16,c_ff=16,dropout=conf['dropout'])

def station_run(g,conf,device,dest,fingerprint,ridge):
    sid=g.iloc[0]['站点ID'];seedvalue=2025+int(sid[1:]);g=g.sort_values('监测时间').reset_index(drop=True)
    n=len(g);L=conf['seq_len'];times=g['监测时间'];xa=g[XCOL].to_numpy(float);ya=g[YCOL].to_numpy(float)
    endpoints=np.arange(L-1,n);window_indices=np.stack([np.arange(i-L+1,i+1) for i in endpoints])
    assert (window_indices[:,-1]==endpoints).all() and (window_indices.max(axis=1)==endpoints).all()
    assert np.all(np.diff(times.to_numpy())>np.timedelta64(0,'s'))
    assert g['站点ID'].nunique()==1
    wt=times.iloc[endpoints].reset_index(drop=True);fitmask=wt<VAL;valmask=(wt>=VAL)&(wt<TEST);refitmask=wt<TEST;testmask=(wt>=TEST)&(wt<=END)
    assert fitmask.sum()>0 and valmask.sum()>0 and testmask.sum()>0
    assert wt[fitmask].max()<wt[valmask].min() and wt[valmask].max()<wt[testmask].min()
    initialraw=times<VAL;fullraw=times<TEST
    sx=StandardScaler().fit(xa[initialraw]);sy=StandardScaler().fit(ya[initialraw])
    wx=sx.transform(xa)[window_indices];wy=sy.transform(ya)[endpoints]
    xtrain=torch.tensor(wx[fitmask],dtype=torch.float32,device=device);ytrain=torch.tensor(wy[fitmask],dtype=torch.float32,device=device)
    xval=torch.tensor(wx[valmask],dtype=torch.float32,device=device);yval=torch.tensor(wy[valmask],dtype=torch.float32,device=device)
    seed(seedvalue);model=new_model(conf).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=conf['lr']);loss_fn=torch.nn.MSELoss()
    best=float('inf');bestepoch=0;bad=0;history=[];start=time.time()
    for epoch in range(1,conf['max_epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True);p=model(xtrain);loss=loss_fn(p,ytrain);assert torch.isfinite(loss);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():vloss=float(loss_fn(model(xval),yval).item())
        assert np.isfinite(vloss)
        improved=vloss<best
        if improved:best=vloss;bestepoch=epoch;bad=0
        else:bad+=1
        history.append({'阶段':'训练验证选epoch','epoch':epoch,'训练标准化MSE':float(loss.item()),'验证标准化MSE':vloss,'验证改善':int(improved)})
        if bad>=conf['patience']:break
    assert bestepoch>0
    # Discard selection model; reinitialize and refit on all pre-test observations.
    del model,optimizer
    sx_final=StandardScaler().fit(xa[fullraw]);sy_final=StandardScaler().fit(ya[fullraw])
    wx_final=sx_final.transform(xa)[window_indices];wy_final=sy_final.transform(ya)[endpoints]
    xr=torch.tensor(wx_final[refitmask],dtype=torch.float32,device=device);yr=torch.tensor(wy_final[refitmask],dtype=torch.float32,device=device)
    seed(seedvalue);final=new_model(conf).to(device);optimizer=torch.optim.Adam(final.parameters(),lr=conf['lr'])
    for epoch in range(1,bestepoch+1):
        final.train();optimizer.zero_grad(set_to_none=True);loss=loss_fn(final(xr),yr);assert torch.isfinite(loss);loss.backward();optimizer.step()
        history.append({'阶段':'前20天重新拟合','epoch':epoch,'训练标准化MSE':float(loss.item()),'验证标准化MSE':np.nan,'验证改善':0})
    # Final test is evaluated only after the validation choice and all fitting are locked.
    final.eval();xt=torch.tensor(wx_final[testmask],dtype=torch.float32,device=device)
    with torch.no_grad():pz=final(xt).cpu().numpy()
    raw=sy_final.inverse_transform(pz);pred=np.maximum(raw,0);truth=ya[endpoints[testmask]]
    assert np.isfinite(pred).all() and np.isfinite(raw).all()
    meanbase=np.tile(ya[fullraw].mean(0),(len(pred),1));lastbase=np.tile(ya[fullraw][-1],(len(pred),1))
    mse_scaled=float(np.mean((pz-wy_final[testmask])**2));meta=g.iloc[0];metricrows=[]
    for j,target in enumerate(YCOL):
        a=score(truth[:,j],pred[:,j]);b=score(truth[:,j],meanbase[:,j]);c=score(truth[:,j],lastbase[:,j]);rr=ridge.loc[(sid,target)]
        assert len(pred)==int(rr['测试记录数'])
        metricrows.append({'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面名称'],'目标变量':target,'历史观测次数':L,'选epoch训练窗口数':int(fitmask.sum()),'验证窗口数':int(valmask.sum()),'重拟合窗口数':int(refitmask.sum()),'重拟合原始观测数':int(fullraw.sum()),'测试窗口数':len(pred),'测试少于20条':int(len(pred)<20),'最优epoch':bestepoch,'选epoch实际训练轮数':history.index(next(h for h in history if h['阶段']=='前20天重新拟合')),'验证最小标准化MSE_四目标平均':best,**a,'未截断MAE':score(truth[:,j],raw[:,j])['MAE'],'原始负预测次数':int((raw[:,j]<0).sum()),'训练均值基线MAE':b['MAE'],'末次训练值基线MAE':c['MAE'],'岭回归MAE':float(rr['MAE']),'岭回归RMSE':float(rr['RMSE']),'岭回归R2':float(rr['R2']),'比岭回归MAE改善比例':1-a['MAE']/float(rr['MAE']) if float(rr['MAE'])>0 else np.nan,'测试实测标准差':float(truth[:,j].std()),'R2可计算':int(np.isfinite(a['R2']))})
    pframe=g.iloc[endpoints[testmask]][['站点ID','省份','流域','断面名称','监测时间']+XCOL].copy();pframe['输入窗口起点']=times.iloc[window_indices[testmask,0]].to_numpy();pframe['输入窗口跨度小时']=(pframe['监测时间'].to_numpy()-pframe['输入窗口起点'].to_numpy())/np.timedelta64(1,'h')
    for j,target in enumerate(YCOL):
        for name,values in [('实测',truth),('预测',pred),('未截断预测',raw),('训练均值基线',meanbase),('末次训练值基线',lastbase)]:pframe[f'{target}_{name}']=values[:,j]
    save(pd.DataFrame(metricrows),dest/'metrics.csv');save(pframe,dest/'predictions.csv');save(pd.DataFrame(history),dest/'history.csv')
    weights={k:v.detach().cpu() for k,v in final.state_dict().items()};torch.save({'state_dict':weights,'config':conf,'station':sid,'seed':seedvalue,'best_epoch':bestepoch,'x_scaler':scaler_record(sx_final),'y_scaler':scaler_record(sy_final),'XCOL':XCOL,'YCOL':YCOL},dest/'model.pt')
    metadata={'station':sid,'fingerprint':fingerprint,'seed':seedvalue,'best_epoch':bestepoch,'validation_best_standardized_mse':best,'test_standardized_mse_four_targets':mse_scaled,'epochs_ran_selection':sum(r['阶段']=='训练验证选epoch' for r in history),'epochs_ran_refit':bestepoch,'n_raw_observations':n,'n_selection_train_windows':int(fitmask.sum()),'n_validation_windows':int(valmask.sum()),'n_refit_windows':int(refitmask.sum()),'n_test_windows':len(pred),'selection_x_scaler':scaler_record(sx),'selection_y_scaler':scaler_record(sy),'final_x_scaler':scaler_record(sx_final),'final_y_scaler':scaler_record(sy_final),'selection_scaler_latest_time':str(times[initialraw].max()),'final_scaler_latest_time':str(times[fullraw].max()),'test_start':str(wt[testmask].min()),'test_end':str(wt[testmask].max()),'n_window_coldstart_dropped':L-1,'seconds':round(time.time()-start,3)}
    dump(dest/'done.json',metadata)
    return pd.DataFrame(metricrows),pframe,metadata

def aggregate(g,basin,target):
    r=g['R2'].dropna();rr=g['岭回归R2'].dropna()
    return {'流域':basin,'目标变量':target,'站点数':len(g),'R2可计算站点数':len(r),'XLinear_MAE中位数':float(g['MAE'].median()),'XLinear_RMSE中位数':float(g['RMSE'].median()),'XLinear_R2中位数':float(r.median()) if len(r) else np.nan,'XLinear_R2_P25':float(r.quantile(.25)) if len(r) else np.nan,'XLinear_R2_P75':float(r.quantile(.75)) if len(r) else np.nan,'岭回归MAE中位数':float(g['岭回归MAE'].median()),'岭回归R2中位数':float(rr.median()) if len(rr) else np.nan,'训练均值基线MAE中位数':float(g['训练均值基线MAE'].median()),'末次训练值基线MAE中位数':float(g['末次训练值基线MAE'].median()),'XLinear_MAE低于岭回归站点数':int((g['MAE']<g['岭回归MAE']).sum()),'XLinear_MAE低于岭回归比例':float((g['MAE']<g['岭回归MAE']).mean()),'测试不足20条站点数':int(g['测试少于20条'].sum()),'说明':'比较优于基线的比例不等于推导成功率'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--patience',type=int,default=10);parser.add_argument('--output',default='results');parser.add_argument('--cpu',action='store_true');args=parser.parse_args()
    out=HERE/args.output;out.mkdir(exist_ok=True);(out/'stations').mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    conf={'seq_len':6,'d_model':16,'dropout':.1,'lr':.001,'max_epochs':args.epochs,'patience':args.patience,'inputs':XCOL,'targets':YCOL,'val_start':str(VAL),'test_start':str(TEST),'test_end':str(END),'seed_base':2025,'refit_pretest':True,'model':'XLinear backbone soft-sensor adaptation'}
    provenance={'source_dataset_sha256':sha(SOURCE),'adapter_sha256':sha(HERE/'adapter.py'),'vendor_sha256':sha(HERE/'vendor_XLinear.py'),'runner_sha256':sha(__file__)};fingerprint=hashlib.sha256(json.dumps({'config':conf,**provenance},sort_keys=True).encode()).hexdigest()
    dump(out/'config.json',{'config':conf,'fingerprint':fingerprint,**provenance})
    dump(out/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'platform':platform.platform()})
    df=pd.read_csv(SOURCE);df['监测时间']=pd.to_datetime(df['监测时间']);assert len(df)==79940 and df['站点ID'].nunique()==750 and np.isfinite(df[XCOL+YCOL].to_numpy(float)).all()
    assert not df.duplicated(['站点ID','监测时间']).any()
    ridge=pd.read_csv(RIDGE).set_index(['站点ID','目标变量']);groups=list(df.groupby('站点ID'));groups=groups[:args.limit] if args.limit else groups
    allm=[];allp=[];allmeta=[];started=time.time()
    for i,(sid,g) in enumerate(groups):
        dest=out/'stations'/sid;dest.mkdir(exist_ok=True)
        if (dest/'done.json').exists():
            meta=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert meta['fingerprint']==fingerprint,('configuration changed; choose a new output directory',sid)
            m=pd.read_csv(dest/'metrics.csv');p=pd.read_csv(dest/'predictions.csv')
        else:m,p,meta=station_run(g,conf,device,dest,fingerprint,ridge)
        assert len(m)==4;allm.append(m);allp.append(p);allmeta.append(meta)
        if i%25==0 or i+1==len(groups):
            elapsed=time.time()-started;print(f'Completed {i+1}/{len(groups)} stations; elapsed {elapsed:.1f}s; last epochs={meta["best_epoch"]}, last station={meta["seconds"]}s',flush=True)
            dump(out/'progress.json',{'completed':i+1,'total':len(groups),'elapsed_seconds':round(elapsed,2),'last_station':sid})
    M=pd.concat(allm,ignore_index=True);P=pd.concat(allp,ignore_index=True);assert len(M)==4*len(groups)
    assert not P.duplicated(['站点ID','监测时间']).any() and np.isfinite(P.select_dtypes(include=[np.number]).to_numpy()).all()
    assert (M['RMSE']+1e-10>=M['MAE']).all()
    save(M,out/'01_逐站点四目标误差.csv');save(P,out/'02_测试实测与预测.csv')
    A=pd.DataFrame([aggregate(g,'全部流域',target) for target,g in M.groupby('目标变量',sort=False)]);B=pd.DataFrame([aggregate(g,basin,target) for (basin,target),g in M.groupby(['流域','目标变量'])]);save(A,out/'03_总体与岭回归对比.csv');save(B,out/'04_分流域误差与对比.csv')
    save(pd.DataFrame([{k:v for k,v in meta.items() if not isinstance(v,(dict,list))} for meta in allmeta]),out/'05_逐站点训练与时间划分.csv')
    # Independently reconstruct all saved metrics from prediction CSV.
    saved=pd.read_csv(out/'02_测试实测与预测.csv');lookup=M.set_index(['站点ID','目标变量'])
    for sid,g in saved.groupby('站点ID'):
        for target in YCOL:
            sc=score(g[target+'_实测'].to_numpy(),g[target+'_预测'].to_numpy());r=lookup.loc[(sid,target)]
            for k in ['MAE','RMSE','R2','偏差']:assert np.isclose(sc[k],r[k],atol=1e-7,rtol=1e-6,equal_nan=True),(sid,target,k)
    if not args.limit:assert len(M)==3000 and len(P)==27326
    assert sha(SOURCE)==provenance['source_dataset_sha256']
    summary={'站点数':len(groups),'目标指标评价数量':len(M),'测试记录数':len(P),'参数量':sum(p.numel() for p in new_model(conf).parameters()),'总体结果':A.to_dict('records'),'原始化学值只用于训练标签与评价':True,'测试输入通道数':5,'历史观测次数':6,'常量或近常量测试目标R2不可计算数量':int(M['R2'].isna().sum()),'测试不足20条站点数':int((M.drop_duplicates('站点ID')['测试少于20条']==1).sum()),'总训练耗时秒':round(time.time()-started,2),'验证':{'同站滑窗不跨站且包含当前易测记录':True,'无未来观测进入输入':True,'训练和验证早于测试':True,'选择与最终scaler仅各自训练期拟合':True,'最终模型按验证选择epoch重初始化并前20天重拟合':True,'未用测试成绩选参数':True,'保存预测复算全部误差一致':True,'与岭回归测试样本一致':True,'原始数据未改动':True},'成功率':'尚无验收阈值，不计算；优于岭回归的比例不是成功率','fingerprint':fingerprint}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
