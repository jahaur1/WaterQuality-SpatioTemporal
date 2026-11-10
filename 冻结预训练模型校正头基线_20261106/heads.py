"""Shared linear residual heads. All choices use pre-test labels only."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
LAMBDAS=[.0001,.001,.01,.1,1.,10.]
def save(df,name):df.to_csv(HERE/name,index=False,encoding='utf-8-sig',float_format='%.12g')
def dump(name,x):(HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def weights(ids):
    _,inv,n=np.unique(ids,return_inverse=True,return_counts=True)
    return 1./n[inv]
def objective(y,p,ids):
    w=weights(ids);return float(np.sum(w*np.mean((y-p)**2,axis=1))/w.sum())
def fit(x,y,ids,lam):
    w=weights(ids);m=Ridge(alpha=lam*w.sum(),fit_intercept=True,solver='svd');m.fit(x,y,sample_weight=w);return m
def metrics(y,p):
    e=p-y;sst=np.sum((y-y.mean())**2);tol=1e-12*max(1.,np.sum(y*y))
    return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),R2=float(1-np.sum(e*e)/sst) if len(y)>=2 and sst>tol else np.nan)
def normalize(z,stats,ids,stage,base=None):
    mu=np.array([stats[sid][stage]['mean'] for sid in ids]);sd=np.array([stats[sid][stage]['scale'] for sid in ids])
    y=(z['truth']-mu[:,5:])/sd[:,5:];last=(z['last_chemical']-mu[:,5:])/sd[:,5:];f=(z['future_easy']-mu[:,:5])/sd[:,:5]
    b=(base-mu[:,5:])/sd[:,5:] if base is not None else None
    return dict(y=y,last=last,f=f,base=b,mu=mu[:,5:],sd=sd[:,5:])
def features(norm,kind):
    if kind=='A_仅预测残差头':return norm['base'],norm['base']
    if kind=='B_结合观测残差头':return np.concatenate([norm['base'],norm['last'],norm['f']],axis=1),norm['base']
    if kind=='观测线性对照':return np.concatenate([norm['last'],norm['f']],axis=1),norm['last']
    raise ValueError(kind)
def main():
    idx=pd.read_csv(HERE/'配对索引.csv');ids=idx['站点ID'].to_numpy();stats=json.loads((HERE/'站点训练标准化.json').read_text(encoding='utf-8'));z=np.load(HERE/'inputs.npz')
    masks={c:idx[c].to_numpy(bool) for c in ['selection_train','validation','refit_train','test']};tr,va,rf,te=[masks[c] for c in masks]
    assert pd.to_datetime(idx.loc[rf,'目标时间']).max()<pd.to_datetime(idx.loc[te,'起报时间']).min()
    ti=np.flatnonzero(te);test=idx.loc[te].reset_index(drop=True);truth=z['truth'][te];predictions={};audit=[];trained={}
    configs=[(b,k) for b in ['Timestral','Chronos2'] for k in ['A_仅预测残差头','B_结合观测残差头']]+[('无预训练模型','观测线性对照')]
    for backbone,kind in configs:
        b=np.load(HERE/backbone/'predictions.npz')['raw'].astype(float) if backbone!='无预训练模型' else None
        ns=normalize(z,stats,ids,'selection',b);nf=normalize(z,stats,ids,'final',b);xs,anchor=features(ns,kind);xf,af=features(nf,kind)
        records=[]
        for lam in LAMBDAS:
            m=fit(xs[tr],(ns['y']-anchor)[tr],ids[tr],lam)
            err=objective(ns['y'][va],anchor[va]+m.predict(xs[va]),ids[va]);records.append({'lambda':lam,'validation_station_equal_standardized_MSE':err})
        best=min(records,key=lambda r:r['validation_station_equal_standardized_MSE']);lam=best['lambda']
        final=fit(xf[rf],(nf['y']-af)[rf],ids[rf],lam)
        raw=(af[te]+final.predict(xf[te]))*nf['sd'][te]+nf['mu'][te]
        name=backbone+'+'+kind if backbone!='无预训练模型' else kind
        predictions[name]=raw
        record={'model':name,'backbone':backbone,'kind':kind,'trainable_parameters':int(final.coef_.size+final.intercept_.size),'selected_lambda':lam,'selection':records,'coefficients':final.coef_.tolist(),'intercept':final.intercept_.tolist(),'selection_train_pairs':int(tr.sum()),'validation_pairs':int(va.sum()),'refit_train_pairs':int(rf.sum()),'test_pairs':int(te.sum()),'fit_latest_target_time':str(idx.loc[rf,'目标时间'].max()),'test_earliest_origin_time':str(idx.loc[te,'起报时间'].min()),'features':(['base_predictions_chemical4'] if kind.startswith('A_') else ['base_predictions_chemical4','origin_chemical4','endpoint_easy5'] if kind.startswith('B_') else ['origin_chemical4','endpoint_easy5']),'test_labels_used_for_fit_or_selection':False}
        audit.append(record);trained[name]=record;print(name,'lambda',lam,'parameters',record['trainable_parameters'],flush=True)
    for backbone in ['Timestral','Chronos2']:
        predictions[backbone+'_冻结原模型']=np.load(HERE/backbone/'predictions.npz')['raw'][te].astype(float)
    predictions['持久性']=z['last_chemical'][te]
    np.savez_compressed(HERE/'test_predictions.npz',truth=truth,test_indices=ti,**predictions)
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'heads.py',HERE/'inputs.npz',HERE/'配对索引.csv',HERE/'站点训练标准化.json']}
    dump('训练配置与头权重.json',{'family':'shared affine residual ridge, offline adaptation only; not full delta-Adapter or COSA reproduction','fingerprints':fingerprints,'lambda_grid':LAMBDAS,'objective':'station-equal mean squared standardized residual error plus lambda coefficient L2 penalty; intercept unpenalized','normalization':'per-station training-only column mean/std; selection raw time<Apr24, final<Apr29','station_scope':'shared across all 750 existing stations; not unseen-station generalization','negative_prediction_postprocess':'clip to zero for every arm; raw saved','models':audit})
    rows=[];summary=[];counts=[];gains=[]
    groups=test.groupby('站点ID').indices
    for name,raw in predictions.items():
        pred=np.maximum(raw,0)
        for sid,ix in groups.items():
            meta=test.iloc[ix[0]]
            for j,target in enumerate(CHEM):rows.append({'模型':name,'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面'],'目标变量':target,'测试配对数':len(ix),**metrics(truth[ix,j],pred[ix,j])})
    M=pd.DataFrame(rows);save(M,'01_逐站点模型误差.csv')
    for (name,target),g in M.groupby(['模型','目标变量'],sort=False):
        summary.append({'模型':name,'目标变量':target,'有效R2站点数':int(g.R2.notna().sum()),'R2均值':g.R2.mean(),'R2中位数':g.R2.median(),'MAE中位数':g.MAE.median(),'RMSE中位数':g.RMSE.median(),'R2大于0.5站点数':int(g.R2.gt(.5).sum()),'R2大于0.8站点数':int(g.R2.gt(.8).sum())})
    for name,g in M.groupby('模型',sort=False):
        counts.append({'模型':name,'站点数':g['站点ID'].nunique(),'任一项R2大于0.5站点数':g.loc[g.R2.gt(.5),'站点ID'].nunique(),'任一项R2大于0.8站点数':g.loc[g.R2.gt(.8),'站点ID'].nunique(),'四项R2均大于0.5站点数':int(g.assign(ok=g.R2.gt(.5)).groupby('站点ID').ok.sum().eq(4).sum()),'测试至少20对且任一项R2大于0.8站点数':g.loc[g.R2.gt(.8)&g['测试配对数'].ge(20),'站点ID'].nunique()})
    for backbone in ['Timestral','Chronos2']:
        base=M[M['模型'].eq(backbone+'_冻结原模型')]
        for kind in ['A_仅预测残差头','B_结合观测残差头']:
            name=backbone+'+'+kind;v=M[M['模型'].eq(name)].merge(base,on=['站点ID','目标变量'],suffixes=('_校正','_原始'),validate='one_to_one')
            v['R2提高量']=v.R2_校正-v.R2_原始;v['MAE降低量']=v.MAE_原始-v.MAE_校正;v['模型']=name;gains.append(v)
    A=pd.DataFrame(summary);C=pd.DataFrame(counts);save(A,'02_逐指标汇总.csv');save(C,'03_高R2站点统计.csv');save(pd.concat(gains),'04_校正相对原模型逐站点收益.csv')
    basin=M.groupby(['模型','流域','目标变量'],sort=False).agg(站点数=('站点ID','nunique'),R2均值=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median')).reset_index();save(basin,'05_分流域汇总.csv')
    overall=A.groupby('模型').agg(四指标等权平均R2=('R2均值','mean')).join(C.set_index('模型'));save(overall.reset_index(),'06_总体比较.csv')
    print(overall.to_string());print(A.to_string(index=False))
if __name__=='__main__':main()
