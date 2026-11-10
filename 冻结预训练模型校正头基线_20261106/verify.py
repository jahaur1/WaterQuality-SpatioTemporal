"""Independent raw-source/scaler and closed-form ridge verification."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    start=time.time();cfg=json.loads((HERE/'训练配置与头权重.json').read_text(encoding='utf-8'))
    for name,digest in cfg['fingerprints'].items():assert sha(HERE/name)==digest,name
    z=np.load(HERE/'inputs.npz');idx=pd.read_csv(HERE/'配对索引.csv');stats=json.loads((HERE/'站点训练标准化.json').read_text(encoding='utf-8'));allid=idx['站点ID'].to_numpy()
    df=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv');df['监测时间']=pd.to_datetime(df['监测时间']);cols=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)','高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];chem=cols[5:]
    h=z['history'];y=z['truth'];f=z['future_easy'];last=z['last_chemical'];count=0
    for sid,g in df.groupby('站点ID'):
        g=g.sort_values('监测时间');raw=g[cols].to_numpy(float);ts=g['监测时间'].to_numpy();ix=np.flatnonzero(allid==sid);q=idx.iloc[ix];oi=q['起报站内索引'].to_numpy(int);ti=q['目标站内索引'].to_numpy(int)
        assert np.array_equal(y[ix],raw[ti,5:]) and np.array_equal(last[ix],raw[oi,5:]) and np.allclose(f[ix],raw[ti,:5],rtol=1e-6,atol=1e-7)
        assert np.array_equal(ts[ti]-ts[oi],np.full(len(ix),np.timedelta64(24,'h')))
        for k,i in zip(ix,oi):
            grid=ts[i]-np.arange(31,-1,-1)*np.timedelta64(4,'h');valid=grid>=ts[i-5];loc=np.searchsorted(ts[i-5:i+1],grid[valid],side='right')-1
            expected=np.full((32,9),np.nan,np.float32);expected[valid]=raw[i-5+loc]
            assert np.array_equal(expected,h[k],equal_nan=True)
        for stage,cut in [('selection','2025-04-24'),('final','2025-04-29')]:
            sc=StandardScaler().fit(raw[ts<np.datetime64(cut)]);r=stats[sid][stage];assert r['n_samples']==sc.n_samples_seen_;assert np.allclose(sc.mean_,r['mean'],rtol=1e-12,atol=1e-10);assert np.allclose(sc.scale_,r['scale'],rtol=1e-10,atol=1e-12);count+=1
    tr=idx.selection_train.to_numpy(bool);va=idx.validation.to_numpy(bool);rf=idx.refit_train.to_numpy(bool);te=idx.test.to_numpy(bool)
    tt=pd.to_datetime(idx['目标时间']);ot=pd.to_datetime(idx['起报时间']);assert tt[tr].max()<pd.Timestamp('2025-04-24');assert tt[rf].max()<ot[te].min();assert tt[va].max()<ot[te].min();assert ot[va].min()>=pd.Timestamp('2025-04-24')
    models=np.load(HERE/'test_predictions.npz');testtruth=y[te];assert np.array_equal(models['truth'],testtruth);head_checks=[]
    for record in cfg['models']:
        backbone=record['backbone'];b=np.load(HERE/backbone/'predictions.npz')['raw'].astype(float) if backbone!='无预训练模型' else None
        def make(stage):
            mu=np.array([stats[sid][stage]['mean'] for sid in allid]);sd=np.array([stats[sid][stage]['scale'] for sid in allid]);target=(y-mu[:,5:])/sd[:,5:];lc=(last-mu[:,5:])/sd[:,5:];easy=(f-mu[:,:5])/sd[:,:5]
            if record['kind'].startswith('A_'):base=(b-mu[:,5:])/sd[:,5:];x=base
            elif record['kind'].startswith('B_'):base=(b-mu[:,5:])/sd[:,5:];x=np.concatenate((base,lc,easy),axis=1)
            else:base=lc;x=np.concatenate((lc,easy),axis=1)
            return x,target-base,base,mu[:,5:],sd[:,5:]
        def closed(x,target,mask,lam):
            _,inv,n=np.unique(allid[mask],return_inverse=True,return_counts=True);w=1./n[inv];xx=x[mask];yy=target[mask];xm=np.average(xx,axis=0,weights=w);ym=np.average(yy,axis=0,weights=w);xc=xx-xm;yc=yy-ym
            co=np.linalg.solve(xc.T@(w[:,None]*xc)+lam*w.sum()*np.eye(xc.shape[1]),xc.T@(w[:,None]*yc));inter=ym-xm@co
            return co,inter
        x,r,anchor,mu,sd=make('selection');scores=[]
        for lam in cfg['lambda_grid']:
            co,inter=closed(x,r,tr,lam);error=np.mean((r[va]-x[va]@co-inter)**2,axis=1);s=pd.DataFrame({'station':allid[va],'error':error}).groupby('station').error.mean().mean();scores.append(s)
        assert np.allclose(scores,[v['validation_station_equal_standardized_MSE'] for v in record['selection']],rtol=1e-8,atol=1e-9)
        assert cfg['lambda_grid'][int(np.argmin(scores))]==record['selected_lambda']
        x,r,anchor,mu,sd=make('final');co,inter=closed(x,r,rf,record['selected_lambda']);assert np.allclose(co.T,record['coefficients'],rtol=1e-7,atol=1e-9);assert np.allclose(inter,record['intercept'],rtol=1e-7,atol=1e-9)
        p=(anchor[te]+x[te]@co+inter)*sd[te]+mu[te];saved=models[record['model']];assert np.allclose(p,saved,rtol=1e-7,atol=1e-8)
        head_checks.append({'model':record['model'],'selection_grid_independently_verified':True,'weighted_ridge_closed_form_coefficients_match':True,'all_test_predictions_replayed':len(p)})
    for backbone,folder in [('Timestral','Timestral209M水质24小时评估_20261104'),('Chronos2','Chronos2水质24小时评估_20261103')]:
        audit=json.loads((HERE/backbone/'完成核验.json').read_text());assert audit['passed'];assert audit['weights_sha256']==sha(ROOT/folder/'weights/model.safetensors')
        pred=np.load(HERE/backbone/'predictions.npz')['raw'];old=np.load(ROOT/folder/'results/future_easy/predictions.npz')['raw'];assert np.array_equal(pred[z['test_new_indices']],old[z['test_old_indices']])
    M=pd.read_csv(HERE/'01_逐站点模型误差.csv');lookup=M.set_index(['模型','站点ID','目标变量']);testids=allid[te];checks=0
    for name in models.files:
        if name in ['truth','test_indices']:continue
        p=np.maximum(models[name],0)
        for sid in np.unique(testids):
            mask=testids==sid
            for j,target in enumerate(chem):
                yy=testtruth[mask,j];pp=p[mask,j];e=pp-yy;sst=np.sum((yy-yy.mean())**2);tol=1e-12*max(1,np.sum(yy*yy));rr=1-np.dot(e,e)/sst if len(yy)>=2 and sst>tol else np.nan;r=lookup.loc[(name,sid,target)]
                assert np.isclose(r.MAE,np.abs(e).mean(),rtol=1e-8,atol=1e-10);assert np.isclose(r.RMSE,np.sqrt(np.mean(e*e)),rtol=1e-8,atol=1e-10);assert np.isclose(r.R2,rr,rtol=1e-8,atol=1e-8,equal_nan=True);checks+=1
    result={'passed':True,'source_windows_verified':len(idx),'stations':750,'scalers_checked':count,'metric_groups_verified':checks,'head_checks':head_checks,'frozen_backbone_weight_hashes_unchanged':True,'previous_test_forecasts_byte_equal':True,'chronological_train_validation_refit_boundaries_verified':True,'elapsed_seconds':time.time()-start}
    (HERE/'独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
