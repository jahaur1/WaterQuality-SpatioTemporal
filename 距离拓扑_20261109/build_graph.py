"""Exploratory geographic graph from explicitly approximate public map locations."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')

def main():
    frame=pd.read_csv(HERE/'站点地图定位.csv').set_index('站点ID')
    nodes=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids']
    frame=frame.reindex(nodes);assert len(frame)==750
    eligible=frame['可用于初步距离图'].fillna(False).astype(str).str.lower().eq('true').to_numpy()
    lon=np.radians(frame['经度'].to_numpy(float));lat=np.radians(frame['纬度'].to_numpy(float))
    delta_lon=lon[:,None]-lon[None,:];delta_lat=lat[:,None]-lat[None,:]
    hav=np.sin(delta_lat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(delta_lon/2)**2
    distance=6371*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))
    basin=frame['流域'].to_numpy();a=np.eye(750);edges=[]
    for i in range(750):
        if not eligible[i]:continue
        candidates=np.flatnonzero(eligible&(basin==basin[i])&(np.arange(750)!=i)&(distance[i]<=250))
        chosen=sorted(candidates,key=lambda j:(distance[i,j],nodes[j]))[:3]
        for j in chosen:
            d=distance[i,j];w=np.exp(-(d/100)**2);a[i,j]=w
            uncertainty=float(frame.iloc[i]['估算半径_km']+frame.iloc[j]['估算半径_km'])
            edges.append([nodes[i],nodes[j],frame.iloc[i]['断面名称'],frame.iloc[j]['断面名称'],d,w,uncertainty,d<=uncertainty])
    a/=a.sum(1,keepdims=True)
    e=pd.DataFrame(edges,columns=['接收站','邻站','接收断面','邻站断面','估算地理距离_km','未归一化权重','合计定位估算半径_km','距离小于两端经验半径之和'])
    e.to_csv(HERE/'距离关联站点.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(HERE/'距离拓扑.npz',adjacency=a,station_ids=nodes,eligible=eligible,distances_km=distance)
    info=dict(stations=750,map_located=int(frame['经度'].notna().sum()),eligible=int(eligible.sum()),connected=int(e['接收站'].nunique()),edges=len(e),self_only=int(np.sum(np.count_nonzero(a,axis=1)==1)),k=3,cutoff_km=250,sigma_km=100,coordinate_system='BD09LL',geographic_approximation=True,uncertainty_overlap_edges=int(e['距离小于两端经验半径之和'].sum()))
    (HERE/'距离建图记录.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(info)

if __name__=='__main__':main()
