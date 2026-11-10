"""Trace public HydroRIVERS connectivity; keep location uncertainty explicit."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'deps'))
import eviltransform
import shapefile
from pyproj import Transformer
from shapely.geometry import LineString,Point,box
from shapely.ops import transform
from shapely.strtree import STRtree
sys.stdout.reconfigure(encoding='utf-8')

def load_network():
    cache=HERE/'river_cache.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=True)
        from shapely import from_wkb
        return pd.DataFrame(z['records'],columns=z['columns']),list(from_wkb(z['wkb']))
    reader=shapefile.Reader(str(HERE/'source/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp'))
    cols=[f[0] for f in reader.fields[1:]]
    records=[];lines=[]
    for sr in reader.iterShapeRecords(bbox=(72,17,137,55)):
        # All Chinese station basins fit inside this buffered extent.
        if len(sr.shape.parts)!=1:
            raise ValueError('Unexpected multipart river reach')
        records.append(list(sr.record));lines.append(LineString(sr.shape.points))
    from shapely import to_wkb
    np.savez_compressed(cache,records=np.array(records),columns=np.array(cols),wkb=to_wkb(lines))
    return pd.DataFrame(records,columns=cols),lines

def main():
    frame=pd.read_csv(ROOT/'距离拓扑_20261109/站点地图定位.csv').set_index('站点ID')
    nodes=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids'].astype(str)
    frame=frame.reindex(nodes)
    records,lines=load_network()
    print('river reaches',len(records),flush=True)
    rid=records.HYRIV_ID.astype(np.int64).to_numpy()
    nxt=records.NEXT_DOWN.astype(np.int64).to_numpy()
    lengths=records.LENGTH_KM.to_numpy(float)
    lookup={int(r):i for i,r in enumerate(rid)}
    # Local azimuthal equidistant projection for station-specific snapping distances.
    tree=STRtree(lines)
    paths={}
    def path(i):
        if i not in paths:
            chain=[];seen=set();current=i;distance=0.
            while current not in seen and distance<=350:
                seen.add(current);chain.append(current);distance+=lengths[current]
                current=lookup.get(int(nxt[current]))
                if current is None:break
            paths[i]=chain
        return paths[i]
    def comparable(a,b):
        return a==b or b in path(a) or a in path(b)
    rows=[];matches={};candidate_rows=[]
    for n,sid in enumerate(nodes):
        entry=frame.loc[sid]
        row=dict(站点ID=sid,省份=entry['省份'],流域=entry['流域'],断面名称=entry['断面名称'],原坐标系='BD09LL',经度_WGS84=np.nan,纬度_WGS84=np.nan,定位估算半径_km=entry['估算半径_km'],河段ID=np.nan,下游河段ID=np.nan,河段等级=np.nan,距河道_km=np.nan,不同支流候选距离差_km=np.nan,保守纳入=False,探索纳入=False,匹配状态='没有合格粗定位',河流名称='官方河网未提供名称',来源='HydroRIVERS v1.0 Asia; 地图粗定位')
        if str(entry['可用于初步距离图']).lower()!='true':
            rows.append(row);continue
        # eviltransform functions take latitude, longitude.
        glat,glon=eviltransform.bd2gcj(float(entry['纬度']),float(entry['经度']))
        lat,lon=eviltransform.gcj2wgs_exact(glat,glon)
        row.update(经度_WGS84=lon,纬度_WGS84=lat)
        # Roundtrip is a coordinate conversion check, not a location accuracy claim.
        b_lat,b_lon=eviltransform.wgs2bd(lat,lon)
        assert eviltransform.distance(b_lat,b_lon,float(entry['纬度']),float(entry['经度']))<5
        project=Transformer.from_crs('EPSG:4326',f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m',always_xy=True).transform
        p=Point(*project(lon,lat))
        candidates=tree.query(box(lon-.3,lat-.3,lon+.3,lat+.3))
        detail=[]
        for k in candidates:
            geom=transform(project,lines[k]);d=geom.distance(p)/1000
            if d<=22:detail.append((d,int(k),geom))
        detail.sort(key=lambda x:(x[0],int(rid[x[1]])))
        if not detail:
            row['匹配状态']='22公里内无河网候选';rows.append(row);continue
        d,i,geom=detail[0]
        other=next((item for item in detail[1:] if not comparable(i,item[1])),None)
        gap=other[0]-d if other is not None else 22-d
        for rank,(cd,ci,cgeom) in enumerate(detail[:8],1):
            candidate_rows.append([sid,rank,int(rid[ci]),int(nxt[ci]),cd,bool(comparable(i,ci))])
        # Determine downstream endpoint using the actual next reach, not line ordering.
        downstream=lookup.get(int(nxt[i]));orientation=None;endpoint_gap=np.nan
        if downstream is not None:
            nextgeom=transform(project,lines[downstream])
            start_d=Point(geom.coords[0]).distance(nextgeom)
            end_d=Point(geom.coords[-1]).distance(nextgeom)
            endpoint_gap=min(start_d,end_d)/1000
            if endpoint_gap<.05 and abs(start_d-end_d)>.1:
                orientation=end_d<start_d
        fraction=geom.project(p,normalized=True)
        remain=(1-fraction if orientation else fraction)*lengths[i] if orientation is not None else np.nan
        radius=float(entry['估算半径_km'])
        # Fixed rules declared without consulting prediction outcomes.
        exploratory=bool(d<=2 and gap>=.5 and orientation is not None)
        conservative=bool(exploratory and radius<=2 and d<=1 and gap>=radius)
        row.update(河段ID=int(rid[i]),下游河段ID=int(nxt[i]),河段等级=int(records.iloc[i].ORD_CLAS),距河道_km=d,不同支流候选距离差_km=gap,保守纳入=conservative,探索纳入=exploratory,匹配状态='自动候选待人工核实' if exploratory else '距离过大或分支歧义或方向未确认',沿河段距下游端_km=remain,河段出口距河口_km=float(records.iloc[i].DIST_DN_KM),几何下游端为末点=orientation,下游衔接误差_km=endpoint_gap)
        matches[n]=(i,remain)
        rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(HERE/'站点河段匹配.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(candidate_rows,columns=['站点ID','候选序号','河段ID','下游河段ID','距河道_km','与首选河段上下游连通']).to_csv(HERE/'河段匹配候选.csv',index=False,encoding='utf-8-sig')
    audits={}
    for label,flag in [('保守','保守纳入'),('探索','探索纳入')]:
        eligible=out[flag].to_numpy(bool)
        a=np.eye(750);edges=[]
        for receiver in np.flatnonzero(eligible):
            r,rr=matches[receiver];candidates=[]
            for source in np.flatnonzero(eligible):
                if source==receiver or frame.iloc[source]['流域']!=frame.iloc[receiver]['流域']:continue
                s,sr=matches[source]
                chain=path(s)
                if r not in chain:continue
                at=chain.index(r)
                # Sum official lengths along the traced downstream route.
                distance=sr-rr if at==0 else sr+sum(lengths[t] for t in chain[1:at])+lengths[r]-rr
                if .1<distance<=250:
                    candidates.append((distance,int(source),at))
            for d,source,hops in sorted(candidates)[:3]:
                w=math.exp(-(d/100)**2)
                a[receiver,source]=w
                edges.append([nodes[source],nodes[receiver],frame.iloc[source]['断面名称'],frame.iloc[receiver]['断面名称'],frame.iloc[receiver]['流域'],int(rid[matches[source][0]]),int(rid[r]),d,hops,w,float(frame.iloc[source]['估算半径_km']),float(frame.iloc[receiver]['估算半径_km'])])
        a/=a.sum(1,keepdims=True)
        np.savez_compressed(HERE/f'河网拓扑_{label}.npz',adjacency=a,station_ids=nodes,eligible=eligible)
        e=pd.DataFrame(edges,columns=['上游站','下游站','上游断面','下游断面','流域','上游河段ID','下游河段ID','沿河距离_km','跨河段数','未归一化权重','上游定位半径_km','下游定位半径_km'])
        e.to_csv(HERE/f'河网关联站点_{label}.csv',index=False,encoding='utf-8-sig')
        audits[label]=dict(matched=int(eligible.sum()),edges=len(edges),receivers=int((np.count_nonzero(a,axis=1)>1).sum()),participating_nodes=len(set(e['上游站'])|set(e['下游站'])),self_only=int((np.count_nonzero(a,axis=1)==1).sum()))
        assert np.allclose(a.sum(1),1)
        assert np.all(np.diag(a)>0)
        assert np.all((a>0)|(a==0))
    audit=dict(source_url='https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip',source_sha256=hashlib.sha256((HERE/'HydroRIVERS_v10_as_shp.zip').read_bytes()).hexdigest(),source_reaches=len(records),coordinate_conversion='BD09LL -> GCJ02 -> approximate WGS84 (eviltransform 0.1.1)',rules=dict(snap_km_max=2,other_branch_distance_gap_min_km=.5,conservative_max_location_radius_km=2,conservative_snap_km_max=1,conservative_gap_min='location radius',upstream_neighbours_max=3,river_distance_max_km=250,sigma_km=100,station_mapping_manually_verified=False,no_travel_time_model=True),graphs=audits)
    (HERE/'建图记录.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
