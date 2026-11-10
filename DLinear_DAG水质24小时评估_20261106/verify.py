"""Independent source/pair/scaler/metric checks plus CPU checkpoint replay; no training."""
import json,hashlib,sys,time,argparse
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from adapter import Conditional24
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=None;MODEL=None
SOURCE=ROOT/'30天_750站点数据集_20250410-20250510'/'水质数据_750站点_20250410-20250510.csv'
EASY=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)'];CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];ALL=EASY+CHEM
SELECT=pd.Timestamp('2025-04-24');VS=pd.Timestamp('2025-04-25');REFIT=pd.Timestamp('2025-04-29');TS=pd.Timestamp('2025-04-30');TE=pd.Timestamp('2025-05-10');H=pd.Timedelta(hours=24)
ARMS=['conditional_future_easy','history_only']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def metrics(y,p):
    e=p-y;sse=np.dot(e,e);yc=y-y.mean();sst=np.dot(yc,yc);tol=1e-12*max(1,float(np.dot(y,y)))
    return {'MAE':np.abs(e).mean(),'RMSE':np.sqrt(np.mean(e*e)),'偏差':e.mean(),'R2':1-sse/sst if len(y)>=2 and sst>tol else np.nan}
def stationdir(sid,arm):
    if MODEL=='DLinear':return HERE/'results_DLinear/stations'/sid/arm
    candidates=list(HERE.glob('results_DAG_shard*/stations/'+sid+'/'+arm));assert len(candidates)==1
    return candidates[0]

def main():
    global OUT,MODEL
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['DLinear','DAG'],required=True);args=ap.parse_args();MODEL=args.model;OUT=HERE/('results_'+MODEL)
    start=time.time();torch.set_num_threads(1);config=readj(OUT/'config.json');summary=readj(OUT/'summary.json')
    assert summary['基础站点数']==750 and summary['站点目标分支指标数']==6000 and summary['全部测试精确配对数']==23136
    for field,path in [('source_dataset_sha256',SOURCE),('runner_sha256',HERE/'run_train.py'),('adapter_sha256',HERE/'adapter.py'),('vendor_sha256',HERE/'vendor_DLinear.py')]:assert sha(path)==config[field],field
    prov=readj(HERE/'source_provenance.json');assert sha(HERE/'source_provenance.json')==config['provenance_sha256']
    for record in prov['files']:
        assert sha(HERE/record['source'])==record['source_sha256'] and sha(HERE/record['vendored'])==record['sha256']
    assert sha(HERE/'vendor_DLinear.py')==sha(HERE/'source_DLinear/models/DLinear.py')
    df=pd.read_csv(SOURCE);df['监测时间']=pd.to_datetime(df['监测时间']);M=pd.read_csv(OUT/'01_逐站点两分支四目标误差.csv');P=pd.read_csv(OUT/'02_精确24小时测试实测与预测.csv')
    for c in ['起报时间','目标时间','输入历史起点']:P[c]=pd.to_datetime(P[c],format='mixed')
    assert len(M)==6000 and len(P)==46272 and M['站点ID'].nunique()==750 and M['可评估'].sum()==6000
    assert not M.duplicated(['站点ID','分支','目标变量']).any();assert not P.duplicated(['站点ID','分支','起报时间','目标时间']).any()
    assert ((P['目标时间']-P['起报时间'])==H).all()
    lookup=M.set_index(['站点ID','分支','目标变量']);predgroups={(sid,arm):g.sort_values('目标时间').reset_index(drop=True) for (sid,arm),g in P.groupby(['站点ID','分支'])}
    stations={sid:g.sort_values('监测时间').reset_index(drop=True) for sid,g in df.groupby('站点ID')};scaler_checks=0;model_groups=0;pair_count=0
    for sid,g in stations.items():
        raw=g[ALL].to_numpy(float);times=g['监测时间'];timeidx={t:i for i,t in enumerate(times)}
        expected=[]
        for i in range(5,len(g)):
            target=times[i]+H
            if TS<=target<=TE and target in timeidx and times[i]>=REFIT:expected.append((i,timeidx[target]))
        oi=np.array([a for a,b in expected]);ti=np.array([b for a,b in expected]);pair_count+=len(expected)
        expected_sel=StandardScaler().fit(raw[(times<SELECT).to_numpy()]);expected_final=StandardScaler().fit(raw[(times<REFIT).to_numpy()])
        for arm in ARMS:
            q=predgroups[sid,arm];assert len(q)==len(expected) and q['可评估'].eq(1).all()
            assert np.array_equal(q['起报站内索引'].to_numpy(),oi) and np.array_equal(q['目标站内索引'].to_numpy(),ti)
            assert np.array_equal(q['目标时间'].to_numpy(),times.iloc[ti].to_numpy()) and np.array_equal(q['起报时间'].to_numpy(),times.iloc[oi].to_numpy())
            assert np.array_equal(q['输入历史起点'].to_numpy(),times.iloc[oi-5].to_numpy())
            for j,c in enumerate(EASY):assert np.allclose(q[c+'_目标时刻易测值'],raw[ti,j],rtol=0,atol=1e-10)
            meta=readj(stationdir(sid,arm)/'done.json');assert meta['fingerprint']==config['fingerprint'] and meta['use_future']==(arm=='conditional_future_easy')
            for name,sc in [('selection_scaler',expected_sel),('final_scaler',expected_final)]:
                sm=meta[name];assert sm['n_samples']==sc.n_samples_seen_;assert np.allclose(sm['mean'],sc.mean_,rtol=1e-10,atol=1e-10);assert np.allclose(sm['scale'],sc.scale_,rtol=1e-10,atol=1e-10);scaler_checks+=1
            assert pd.Timestamp(meta['selection_scaler_latest_time'])<SELECT and pd.Timestamp(meta['final_scaler_latest_time'])<REFIT
            assert pd.Timestamp(meta['selection_latest_label'])<SELECT and pd.Timestamp(meta['validation_latest_label'])<REFIT
            assert pd.Timestamp(meta['final_train_latest_label'])<q['起报时间'].min() and pd.Timestamp(meta['validation_latest_label'])<q['起报时间'].min()
            hist=pd.read_csv(stationdir(sid,arm)/'history.csv');sel=hist[hist['阶段']=='选择epoch'];ref=hist[hist['阶段']=='起报前重新拟合'];bestrow=sel.loc[sel['验证标准化MSE'].idxmin()]
            assert int(bestrow['epoch'])==meta['best_epoch'] and len(ref)==meta['best_epoch'] and len(sel)==meta['epochs_selection']
            for j,target in enumerate(CHEM):
                y=q[target+'_实测'].to_numpy();p=q[target+'_预测'].to_numpy();persist=q[target+'_起报时化学值持久性'].to_numpy();mean=q[target+'_训练均值基线'].to_numpy()
                assert np.allclose(y,raw[ti,j+5],atol=1e-10,rtol=0) and np.allclose(persist,raw[oi,j+5],atol=1e-10,rtol=0)
                assert np.allclose(mean,raw[(times<REFIT).to_numpy(),j+5].mean(),atol=1e-9,rtol=1e-9)
                assert np.isfinite(p).all() and (p>=0).all();r=lookup.loc[(sid,arm,target)]
                for values,prefix in [(p,''),(persist,'起报时化学值持久性_'),(mean,'训练均值基线_')]:
                    sm=metrics(y,values)
                    for k,v in sm.items():assert np.isclose(v,r[prefix+k],rtol=1e-6,atol=1e-7,equal_nan=True),(sid,arm,target,prefix+k)
                model_groups+=1
        a=predgroups[sid,ARMS[0]];b=predgroups[sid,ARMS[1]]
        assert np.array_equal(a['目标时间'].to_numpy(),b['目标时间'].to_numpy())
    assert pair_count==23136 and scaler_checks==3000 and model_groups==6000
    ids=sorted(stations);chosen=[ids[k] for k in [0,187,374,561,749]];replays=[]
    for sid in chosen:
        g=stations[sid];raw=g[ALL].to_numpy(float)
        for arm in ARMS:
            q=predgroups[sid,arm];checkpoint=torch.load(stationdir(sid,arm)/'model.pt',map_location='cpu',weights_only=True)
            scaler=checkpoint['scaler'];mu=np.array(scaler['mean']);scale=np.array(scaler['scale']);oi=q['起报站内索引'].to_numpy(int);ti=q['目标站内索引'].to_numpy(int);hidx=np.stack([np.arange(i-5,i+1) for i in oi])
            hs=torch.tensor((raw[hidx]-mu)/scale,dtype=torch.float32);fs=torch.tensor(((raw[ti,:5]-mu[:5])/scale[:5])[:,None,:],dtype=torch.float32)
            conf=checkpoint['config'];future=arm=='conditional_future_easy';model=Conditional24(MODEL,use_future=future).eval();model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                outz=model(hs,fs if future else None).numpy()
                one=model(hs[:1],fs[:1] if future else None).numpy();assert np.allclose(one,outz[:1],rtol=1e-4,atol=1e-5),('batch isolation',MODEL,sid,arm)
                if not future:assert torch.equal(model(hs,None),model(hs,fs.flip(0)))
            rawp=outz*scale[5:]+mu[5:];stored=q[[c+'_未截断预测' for c in CHEM]].to_numpy(float)
            zstored=(stored-mu[5:])/scale[5:];assert np.allclose(outz,zstored,rtol=1e-4,atol=1e-4) and np.allclose(rawp,stored,rtol=1e-4,atol=1e-5)
            replays.append({'station':sid,'arm':arm,'test_pairs':len(q),'max_abs_standardized_difference':float(np.max(np.abs(outz-zstored))),'passed':True})
    result={'passed':True,'station_count':750,'metrics_rows':6000,'unique_test_pairs':pair_count,'two_arm_prediction_rows':len(P),'source_sha256':sha(SOURCE),'audited_scalers':scaler_checks,'independent_metric_groups':model_groups,'exact_24h_no_interpolation':True,'same_pairs_and_truth_between_arms':True,'history_ends_at_origin_and_uses_past_six_same_station_rows':True,'future_covariates_only_target_time_five_easy_values':True,'persistence_baseline_equals_origin_chemical_values':True,'all_training_and_validation_label_times_precede_earliest_test_origin':True,'training_only_scaler_statistics_match_source':True,'official_DLinear_byte_exact_and_DAG_documented_patches_verified':True,'cpu_checkpoint_replay':replays,'elapsed_seconds':round(time.time()-start,3),'limitations':['Future easy measurements are intentionally known retrospective covariates, not real forecasts.','Historical chemical updates at every origin are user-assumed available; reporting delays and reduced testing are not simulated.','Saved data and replay verify outputs; exclusion of final test from model selection is also supported by code review.']}
    (OUT/'独立验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
