"""Learn forecast-oriented delays on training labels, with exact timestamp pairing."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
LAGS=np.array([0,4,8,12,24])

def corr(y,x):
    ok=np.isfinite(y)&np.isfinite(x);n=ok.sum(0)
    yy=np.where(ok,y,0);xx=np.where(ok,x,0)
    sy=yy.sum(0);sx=xx.sum(0)
    vy=(yy**2).sum(0)-sy**2/np.maximum(n,1)
    vx=(xx**2).sum(0)-sx**2/np.maximum(n,1)
    cov=(yy*xx).sum(0)-sy*sx/np.maximum(n,1)
    r=np.divide(cov,np.sqrt(np.maximum(vy,0)*np.maximum(vx,0)),out=np.full(len(n),np.nan),where=(vy>1e-12)&(vx>1e-12)&(n>=3))
    return np.clip(r,-1,1),n

def align(frame,times,receiver,sender,lag):
    yi=frame.index.get_indexer(times)
    xi=frame.index.get_indexer(times-pd.Timedelta(hours=24+int(lag)))
    v=frame.to_numpy(float)
    y=v[yi[:,None],receiver[None,:]]
    x=v[xi.clip(0)[:,None],sender[None,:]].copy();x[xi<0]=np.nan
    return y,x

def build(stage,cut,raw,nodes):
    e=pd.read_csv(ROOT/f'水质跨站预测_20261109/{stage}_关联站点.csv')
    node_index={s:i for i,s in enumerate(nodes)}
    selected=[];checks=[];support=np.zeros((len(LAGS),4,750,750))
    zero=np.zeros_like(support)
    for c,chemical in enumerate(CHEM):
        edges=e[e['化学指标'].eq(chemical)].copy().reset_index(drop=True)
        receiver=edges['接收站'].map(node_index).to_numpy();sender=edges['邻站'].map(node_index).to_numpy()
        frame=raw.pivot(index='监测时间',columns='站点ID',values=chemical).reindex(columns=nodes).sort_index()
        train=frame.index[frame.index<pd.Timestamp(cut)]
        rs=[];ns=[]
        for lag in LAGS:
            y,x=align(frame,train,receiver,sender,lag)
            r,n=corr(y,x);rs.append(r);ns.append(n)
        rs=np.array(rs);ns=np.array(ns)
        score=rs*ns/(ns+16.)
        score[(ns<16)|~np.isfinite(score)]=-np.inf
        best=score.argmax(0)
        # Require a positive delayed relationship and >0.02 penalized score gain.
        use=(best>0)&(score[best,np.arange(len(edges))]>np.maximum(score[0],0)+.02)
        chosen=np.where(use,best,0)
        for i,row in edges.iterrows():
            k=int(chosen[i]);r=int(receiver[i]);s=int(sender[i]);w=float(row['权重'])
            support[k,c,r,s]=w;zero[0,c,r,s]=w
            selected.append(dict(阶段=stage,指标=chemical,接收站=row['接收站'],邻站=row['邻站'],流域=row['流域'],接收断面=row['接收站断面'],邻站断面=row['邻站断面'],选择滞后_小时=int(LAGS[k]),训练配对数=int(ns[k,i]),零滞后训练配对数=int(ns[0,i]),训练Pearson_r=rs[k,i],零滞后训练Pearson_r=rs[0,i],惩罚后得分=score[k,i] if np.isfinite(score[k,i]) else np.nan,原边权重=w,训练标签截止_不含=cut))
        if stage=='selection':
            validation=frame.index[(frame.index>=pd.Timestamp('2025-04-25'))&(frame.index<pd.Timestamp('2025-04-29'))]
            y,x0=align(frame,validation,receiver,sender,0)
            xs=[]
            for lag in LAGS:xs.append(align(frame,validation,receiver,sender,lag)[1])
            chosen_x=np.take_along_axis(np.stack(xs),chosen[None,None,:],axis=0)[0]
            common=np.isfinite(y)&np.isfinite(x0)&np.isfinite(chosen_x)
            r0,n=corr(np.where(common,y,np.nan),np.where(common,x0,np.nan))
            rl,_=corr(np.where(common,y,np.nan),np.where(common,chosen_x,np.nan))
            for i,row in edges.iterrows():
                checks.append([chemical,row['接收站'],row['邻站'],int(LAGS[chosen[i]]),int(n[i]),r0[i] if n[i]>=6 else np.nan,rl[i] if n[i]>=6 else np.nan])
    for c in range(4):
        support[0,c,np.arange(750),np.arange(750)]=1
        zero[0,c,np.arange(750),np.arange(750)]=1
        denominator=zero[:,c].sum((0,2))
        support[:,c]/=denominator[None,:,None];zero[:,c]/=denominator[None,:,None]
    np.testing.assert_allclose(support.sum(0),zero.sum(0),atol=1e-12)
    np.testing.assert_allclose(support.sum((0,3)),1,atol=1e-12)
    np.savez_compressed(HERE/f'{stage}_时滞图.npz',station_ids=nodes,lags=LAGS,lagged=support,zero=zero)
    out=pd.DataFrame(selected);out.to_csv(HERE/f'{stage}_时滞选择.csv',index=False,encoding='utf-8-sig')
    if checks:
        checks=pd.DataFrame(checks,columns=['指标','接收站','邻站','选择滞后_小时','共同验证配对数','验证零滞后r','验证所选滞后r'])
        checks['验证r变化']=checks['验证所选滞后r']-checks['验证零滞后r']
        checks.to_csv(HERE/'内部时间验证_关系稳定性.csv',index=False,encoding='utf-8-sig')
    return dict(cut=cut,latest_target=str(train.max()),latest_possible_source=str(train.max()-pd.Timedelta(hours=24)),edges=len(out),nonzero_lag_edges=int(out['选择滞后_小时'].gt(0).sum()),lag_counts={str(k):int(v) for k,v in out['选择滞后_小时'].value_counts().sort_index().items()})

def main():
    raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间'])
    nodes=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids']
    report={stage:build(stage,cut,raw,nodes) for stage,cut in [('selection','2025-04-24'),('final','2025-04-29')]}
    report['rules']=dict(candidate_lags_hours=LAGS.tolist(),relationship='receiver chemical at origin+24h versus sender same chemical at origin-lag',min_exact_pairs=16,score='r*n/(n+16)',minimum_score_gain=.02,same_edges_and_weights_for_zero_and_lagged=True,negative_lags_forbidden=True,physical_upstream_not_assumed=True)
    (HERE/'时滞建图记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
