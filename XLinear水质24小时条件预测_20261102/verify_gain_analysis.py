from pathlib import Path
from itertools import product
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
HERE=Path(__file__).parent;BASE=HERE/'results';OUT=BASE/'R2提升幅度与显著性'
R=pd.read_csv(OUT/'01_全部750站四指标提升与探索性检验.csv');W=pd.read_csv(OUT/'03_613个提高站点一站一行.csv');S=json.loads((OUT/'分析摘要.json').read_text(encoding='utf-8'))
assert len(R)==3000 and len(W)==613 and R[R['R2数值提高']==1]['站点ID'].nunique()==613 and R['R2数值提高'].sum()==1378
assert np.allclose(R['加入未来易测R2']-R['仅历史R2'],R['R2提高量'],equal_nan=True,rtol=1e-8,atol=1e-9)
for label in ['48h','72h']:
    p=R[label+'_单侧近似p'].to_numpy();finite=np.isfinite(p)
    for method,name in [('bh','BH'),('by','BY')]:
        q=false_discovery_control(np.where(finite,p,1),method=method)
        assert np.allclose(q[finite],R.loc[finite,label+'_'+name+'校正q'],rtol=1e-8,atol=1e-10)
    expected=(R['R2提高量']>0)&(R[label+'_BY校正q']<.05)&(R[label+'_R2差值95区间下限']>0)
    assert np.array_equal(expected.to_numpy(int),R[label+'_探索性显著改善'].to_numpy(int))
assert (W['探索性显著提高指标数']>0).sum()==24
for filename,digest in S['输入文件SHA256'].items():assert hashlib.sha256((BASE/filename).read_bytes()).hexdigest()==digest
# Independently expand all 100,000 draws for one hinted comparison.
example=R[R['48h_探索性显著改善']==1].iloc[0];sid=example['站点ID'];target=example['化学指标'];P=pd.read_csv(BASE/'02_精确24小时测试实测与预测.csv');P=P[P['站点ID']==sid]
a=P[P['分支']=='conditional_future_easy'].copy();b=P[P['分支']=='history_only'].copy()
for f in [a,b]:f['目标时间']=pd.to_datetime(f['目标时间'],format='mixed')
a=a.sort_values('目标时间');b=b.sort_values('目标时间');assert a['目标时间'].to_list()==b['目标时间'].to_list()
y=a[target+'_实测'].to_numpy();d=(y-b[target+'_预测'].to_numpy())**2-(y-a[target+'_预测'].to_numpy())**2;day=np.minimum(((a['目标时间']-pd.Timestamp('2025-04-30')).dt.total_seconds()//86400).to_numpy(int),9)
count=np.bincount(day,minlength=10);loss=np.bincount(day,weights=d,minlength=10);sy=np.bincount(day,weights=y,minlength=10);sy2=np.bincount(day,weights=y*y,minlength=10)
starts=np.array(list(product(range(10),repeat=5)));idx=np.stack([starts,(starts+1)%10],axis=2).reshape(100000,10)
nn=count[idx].sum(1);valid=nn>1;nullsum=(loss-d.mean()*count)[idx].sum(1)
p=(1+np.count_nonzero(nullsum[valid]/nn[valid]>=d.mean()))/(1+valid.sum());assert np.isclose(p,example['48h_单侧近似p'],rtol=1e-10,atol=1e-12)
sumy=sy[idx].sum(1);sumy2=sy2[idx].sum(1);sst=np.full(100000,np.nan);sst[valid]=sumy2[valid]-sumy[valid]**2/nn[valid]
good=valid&(sst>1e-12*np.maximum(1.,np.abs(sumy2)));delta=loss[idx].sum(1)[good]/sst[good]
lo,hi=np.quantile(delta,[.025,.975],method='inverted_cdf');assert np.allclose([lo,hi],[example['48h_R2差值95区间下限'],example['48h_R2差值95区间上限']],rtol=1e-6,atol=1e-8)
audit={'passed':True,'metric_pairs':3000,'improved_stations':613,'positive_gain_pairs':1378,'48h_hint_stations':24,'q_values_crosschecked_with_scipy_BH_and_BY':True,'all_flags_match_declared_criteria':True,'original_predictions_and_metrics_unchanged':True,'uncompressed_100000_resample_check':{'station':sid,'target':target,'smoothed_p':float(p),'gain_CI95':[float(lo),float(hi)],'matches_compressed_weight_implementation':True},'meaning':'Computational verification only; does not establish the validity of 10-day bootstrap inference.'}
(OUT/'独立计算核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit,ensure_ascii=True,indent=2))
