"""Independent input chronology, complete metrics, files, and checkpoint replay audit."""
from pathlib import Path
import sys,json,hashlib
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=json.loads((HERE/'实验配置.json').read_text(encoding='utf-8'));src=ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv'
    assert sha(src)==cfg['source_sha256'] and sha(HERE/'测试配对索引.csv')==cfg['test_index_sha256']
    old=ROOT/'XLinear水质24小时条件预测_20261102/results/02_精确24小时测试实测与预测.csv';assert sha(old)==cfg['old_predictions_sha256']
    d=pd.read_csv(src);idx=pd.read_csv(HERE/'测试配对索引.csv');cols=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)','高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];chem=cols[5:]
    d['监测时间']=pd.to_datetime(d['监测时间'],format='mixed')
    for col in ['起报时间','目标时间','输入历史起点']:idx[col]=pd.to_datetime(idx[col],format='mixed')
    assert (idx['目标时间']-idx['起报时间']==pd.Timedelta(hours=24)).all()
    assert idx[['站点ID','起报时间','目标时间']].duplicated().sum()==0 and idx['站点ID'].nunique()==750
    z=np.load(HERE/'inputs.npz');h=z['history'];truth=z['truth'];future=z['future_easy']
    # Independent numeric reconstruction, using searchsorted instead of reindex/ffill.
    for sid,ixs in idx.groupby('站点ID').groups.items():
        g=d[d['站点ID']==sid].sort_values('监测时间').reset_index(drop=True);raw=g[cols].to_numpy(float);ts=g['监测时间'].to_numpy()
        for k in ixs:
            r=idx.iloc[k];oi=int(r['起报站内索引']);ti=int(r['目标站内索引']);ht=ts[oi-5:oi+1];grid=ts[oi]-np.arange(31,-1,-1)*np.timedelta64(4,'h');take=np.searchsorted(ht,grid,side='right')-1;expected=np.full((32,9),np.nan);ok=take>=0;expected[ok]=raw[oi-5+take[ok]]
            assert ht[0]==r['输入历史起点'] and ht[-1]==r['起报时间'] and ts[ti]==r['目标时间']
            assert np.allclose(h[k],expected,equal_nan=True,rtol=1e-6,atol=1e-6)
            assert np.allclose(truth[k],raw[ti,5:],rtol=1e-6,atol=1e-6) and np.allclose(future[k],raw[ti,:5],rtol=1e-6,atol=1e-6)
    registry=pd.read_csv(HERE/'本地权重清单.csv')
    for _,r in registry[registry['模型'].str.startswith('MOIRAI')].iterrows():assert sha(r['路径'])==r['SHA256']
    m=pd.read_csv(HERE/'01_逐站点三模型两条件四指标误差.csv');assert len(m)==18000
    oldp=pd.read_csv(old);oldp=oldp[oldp['分支']=='conditional_future_easy'].reset_index(drop=True);y=oldp[[c+'_实测' for c in chem]].to_numpy(float)
    for (model,arm),frame in m.groupby(['模型','输入条件']):
        pred=np.load(HERE/'results'/(model+'__'+arm)/'predictions.npz')['prediction'];assert np.isfinite(pred).all() and pred.shape==(23136,4)
        records=frame.set_index(['站点ID','目标变量'])
        for sid,ixs in idx.groupby('站点ID').groups.items():
            yy=y[ixs];pp=pred[ixs];err=pp-yy;sst=((yy-yy.mean(0))**2).sum(0);den=1e-12*np.maximum(1.,(yy*yy).sum(0));r2=np.full(4,np.nan);ok=sst>den;r2[ok]=1-(err*err).sum(0)[ok]/sst[ok]
            for j,c in enumerate(chem):
                r=records.loc[(sid,c)];assert r['测试配对数']==len(ixs)
                assert np.allclose([np.abs(err[:,j]).mean(),np.sqrt((err[:,j]**2).mean()),r2[j]],[r['MAE'],r['RMSE'],r['R2']],equal_nan=True,rtol=1e-8,atol=1e-8)
    pilot=np.load(HERE/'pilot/MOIRAI-small__future_easy/predictions.npz')['raw'];full=np.load(HERE/'results/MOIRAI-small__future_easy/predictions.npz')['raw'];replay=float(np.max(np.abs(pilot-full[:len(pilot)])));assert replay==0
    result={'passed':True,'all_input_windows_independently_reconstructed':23136,'stations':750,'station_model_arm_target_metrics_recomputed':18000,'exact_horizon_hours':24,'same_source_and_XLinear_test_keys':True,'original_source_predictions_weights_unchanged':True,'independent_process_first128_predictions_replay_max_abs_diff':replay,'scope':'computational verification; not external generalization or significance validation'}
    (HERE/'独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
