"""Constrained DTW graph, matching the Pearson graph's per-indicator degree and mass."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']

def batch_distance(x,y,times,lengths):
    """Minimum cumulative squared cost with a strict 12-hour time band."""
    count,width=x.shape;prev=np.full((count,width+1),np.inf);prev[:,0]=0
    output=np.full(count,np.inf);rows=np.arange(count)
    for i in range(width):
        curr=np.full_like(prev,np.inf)
        for j in range(max(0,i-3),min(width,i+4)):
            allowed=(i<lengths)&(j<lengths)&(np.abs(times[:,i]-times[:,j])<=12)
            value=(x[:,i]-y[:,j])**2+np.minimum(np.minimum(prev[:,j+1],curr[:,j]),prev[:,j])
            curr[:,j+1]=np.where(allowed,value,np.inf)
        ended=lengths==i+1
        output[ended]=curr[rows[ended],lengths[ended]]
        prev=curr
    return np.sqrt(output/lengths)

def distances(frame):
    values=frame.to_numpy(float).T;size=len(values);left,right=np.triu_indices(size,1)
    out=np.full((size,size),np.inf);overlaps=np.zeros((size,size),int)
    hours=(frame.index-frame.index[0]).total_seconds().to_numpy()/3600
    for start in range(0,len(left),1024):
        a=left[start:start+1024];b=right[start:start+1024];common=np.isfinite(values[a])&np.isfinite(values[b]);n=common.sum(1)
        order=np.argsort(~common,axis=1,kind='stable');x=np.take_along_axis(values[a],order,1);y=np.take_along_axis(values[b],order,1);t=hours[order]
        valid=np.arange(frame.shape[0])[None,:]<n[:,None]
        x=np.where(valid,x,0);y=np.where(valid,y,0);den=np.maximum(n,1)
        mx=x.sum(1)/den;my=y.sum(1)/den
        sx=np.sqrt(np.where(valid,(x-mx[:,None])**2,0).sum(1)/den);sy=np.sqrt(np.where(valid,(y-my[:,None])**2,0).sum(1)/den)
        eligible=(n>=16)&(sx>1e-10)&(sy>1e-10);selected=np.flatnonzero(eligible)
        if len(selected):
            xx=(x[selected]-mx[selected,None])/sx[selected,None];yy=(y[selected]-my[selected,None])/sy[selected,None]
            d=batch_distance(xx,yy,t[selected],n[selected]);out[a[selected],b[selected]]=d;out[b[selected],a[selected]]=d
        overlaps[a,b]=n;overlaps[b,a]=n
    return out,overlaps

def main():
    raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间'])
    train=raw[raw['监测时间']<'2025-04-29'];meta=raw.drop_duplicates('站点ID').set_index('站点ID');stations=sorted(meta.index);nodes={s:i for i,s in enumerate(stations)}
    old=pd.read_csv(ROOT/'水质跨站预测_20261109/final_关联站点.csv');edges=[];a=np.eye(750)
    for basin,group in train.groupby('流域'):
        for chem in CHEM:
            frame=group.pivot(index='监测时间',columns='站点ID',values=chem);d,n=distances(frame);ids=frame.columns
            for i,sid in enumerate(ids):
                previous=old[old['接收站'].eq(sid)&old['化学指标'].eq(chem)]
                degree=len(previous)
                if not degree:continue
                selected=sorted(np.flatnonzero(np.isfinite(d[i])),key=lambda j:(d[i,j],ids[j]))[:degree]
                assert len(selected)==degree
                scores=np.array([np.exp(-d[i,j])*n[i,j]/(n[i,j]+16) for j in selected]);weights=scores/scores.sum()*previous['权重'].sum()
                for j,w in zip(selected,weights):
                    a[nodes[sid],nodes[ids[j]]]+=w/4
                    edges.append([basin,chem,sid,ids[j],meta.loc[sid,'断面名称'],meta.loc[ids[j],'断面名称'],int(n[i,j]),d[i,j],w])
        print('Built',basin,flush=True)
    a/=a.sum(1,keepdims=True)
    previous=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['final']
    np.testing.assert_allclose(a.diagonal(),previous.diagonal(),atol=1e-7)
    np.savez_compressed(HERE/'DTW拓扑.npz',adjacency=a,station_ids=stations)
    e=pd.DataFrame(edges,columns=['流域','指标','接收站','邻站','接收断面','邻站断面','同步样本数','标准化DTW距离','权重']);e.to_csv(HERE/'DTW关联站点.csv',index=False,encoding='utf-8-sig')
    changed=np.count_nonzero((a>0)!=(previous>0))
    (HERE/'建图记录.json').write_text(json.dumps(dict(cut='2025-04-29',max_lag_hours=12,min_common=16,indicator_edges=len(e),unique_edges=int(np.count_nonzero(a)-750),changed_binary_entries=changed,matched_degree_and_neighbor_weight_mass=True),indent=2),encoding='utf-8')
    print('indicator edges',len(e),'changed entries',changed,flush=True)

if __name__=='__main__':main()
