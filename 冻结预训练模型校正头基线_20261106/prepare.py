"""Build causal windows for training a shared residual head; no target filling."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(ROOT/'DLinear_DAG水质24小时评估_20261106'))
from run_train import pairs_for,ALL,EASY,CHEM,SOURCE,EXPECTED_SOURCE_SHA
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED_SOURCE_SHA
df=pd.read_csv(SOURCE);df['监测时间']=pd.to_datetime(df['监测时间'])
hs=[];fs=[];ys=[];lasts=[];indices=[];stats={}
for sid,g in df.groupby('站点ID'):
    g,oi,ti,hi,ot,tt,masks,coverage=pairs_for(g)
    raw=g[ALL].to_numpy(float);ts=g['监测时间'].to_numpy();hist=np.full((len(oi),32,9),np.nan,np.float32)
    for k,i in enumerate(oi):
        times=ts[i-5:i+1];grid=ts[i]-np.arange(31,-1,-1)*np.timedelta64(4,'h')
        take=np.searchsorted(times,grid,side='right')-1;ok=take>=0
        hist[k,ok]=raw[i-5+take[ok]]
    hs.append(hist);fs.append(raw[ti,:5]);ys.append(raw[ti,5:]);lasts.append(raw[oi,5:])
    for k,(i,j) in enumerate(zip(oi,ti)):
        indices.append({'站点ID':sid,'省份':g.iloc[0]['省份'],'流域':g.iloc[0]['流域'],'断面':g.iloc[0]['断面名称'],'起报时间':ot.iloc[k],'目标时间':tt.iloc[k],'起报站内索引':i,'目标站内索引':j,**{name:bool(mask[k]) for name,mask in masks.items()}})
    stats[sid]={}
    for name,cut in [('selection','2025-04-24'),('final','2025-04-29')]:
        x=raw[ts<np.datetime64(cut)];sc=StandardScaler().fit(x)
        stats[sid][name]={'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'n_samples':len(x)}
idx=pd.DataFrame(indices);h=np.concatenate(hs);f=np.concatenate(fs);y=np.concatenate(ys);last=np.concatenate(lasts)
ref=ROOT/'预训练模型水质24小时评估_20261103'
testidx=pd.read_csv(ref/'测试配对索引.csv');keys=['站点ID','起报站内索引','目标站内索引']
mapping=idx.reset_index().merge(testidx[keys].reset_index().rename(columns={'index':'old_index'}),on=keys,validate='one_to_one')
assert len(mapping)==23136 and idx.test.sum()==23136
z=np.load(ref/'inputs.npz');ii=mapping['index'].to_numpy();jj=mapping.old_index.to_numpy()
assert np.array_equal(h[ii],z['history'][jj],equal_nan=True)
assert np.allclose(f[ii],z['future_easy'][jj],rtol=1e-6,atol=1e-7)
assert np.allclose(y[ii],z['truth'][jj],rtol=1e-6,atol=1e-7)
np.savez_compressed(HERE/'inputs.npz',history=h,future_easy=f.astype(np.float32),truth=y,last_chemical=last,test_new_indices=ii,test_old_indices=jj)
idx.to_csv(HERE/'配对索引.csv',index=False,encoding='utf-8-sig')
(HERE/'站点训练标准化.json').write_text(json.dumps(stats,ensure_ascii=False),encoding='utf-8')
out={'source_sha256':sha(SOURCE),'total_pairs':len(idx),'stations':idx['站点ID'].nunique(),'splits':{c:int(idx[c].sum()) for c in ['selection_train','validation','refit_train','test']},'test_history_exactly_matches_previous_inputs':True,'historical_chemistry_ends_at_origin':True,'future_easy_only_endpoint':True,'no_future_chemical_input':True}
(HERE/'配对核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out)
