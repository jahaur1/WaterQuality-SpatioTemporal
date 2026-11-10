"""Independent route, distance, coordinate and convolution-direction checks."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE/'deps'))
import eviltransform
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import Point
from shapely.ops import transform
sys.stdout.reconfigure(encoding='utf-8')

cache=np.load(HERE/'river_cache.npz',allow_pickle=True)
records=cache['records'];wkb=cache['wkb']
attrs=pd.DataFrame(records,columns=cache['columns']).set_index('HYRIV_ID')
positions={int(r):i for i,r in enumerate(records[:,0])}
mapping=pd.read_csv(HERE/'站点河段匹配.csv').set_index('站点ID')
original=pd.read_csv(ROOT/'距离拓扑_20261109/站点地图定位.csv').set_index('站点ID')
edges=pd.read_csv(HERE/'河网关联站点_探索.csv')
npz=np.load(HERE/'河网拓扑_探索.npz')
ids=npz['station_ids'];index={s:i for i,s in enumerate(ids)}
a=npz['adjacency'];assert a.shape==(750,750)
assert np.allclose(a.sum(1),1) and np.all(a>=0)
rebuilt=np.eye(750);checked=[]
for e in edges.to_dict('records'):
    s=e['上游站'];r=e['下游站'];sr=int(e['上游河段ID']);rr=int(e['下游河段ID'])
    assert mapping.loc[s,'流域']==mapping.loc[r,'流域']==e['流域']
    current=sr;chain=[current]
    while current!=rr:
        current=int(attrs.loc[current,'NEXT_DOWN'])
        assert current and current in attrs.index and current not in chain
        chain.append(current)
        assert len(chain)<10000
    assert len(chain)-1==e['跨河段数']
    d=float(attrs.loc[sr,'DIST_DN_KM']+mapping.loc[s,'沿河段距下游端_km']-attrs.loc[rr,'DIST_DN_KM']-mapping.loc[r,'沿河段距下游端_km'])
    discrepancy=abs(d-e['沿河距离_km'])
    # Outlet distances and sums of rounded reach lengths are distinct attributes.
    # Check agreement within 1% (minimum 0.2km); keep actual discrepancies below.
    assert discrepancy < max(.2,.01*e['沿河距离_km']),(s,r,d,e['沿河距离_km'])
    # Confirm all stored links really share geometric junction endpoints.
    endpoint_error=0.
    for left,right in zip(chain[:-1],chain[1:]):
        lg=from_wkb(wkb[positions[left]])
        rg=from_wkb(wkb[positions[right]])
        best=min(eviltransform.distance(p[1],p[0],q[1],q[0]) for p in [lg.coords[0],lg.coords[-1]] for q in [rg.coords[0],rg.coords[-1]])
        endpoint_error=max(endpoint_error,best)
        assert best<50
    w=np.exp(-(e['沿河距离_km']/100)**2)
    assert abs(w-e['未归一化权重'])<1e-12
    rebuilt[index[r],index[s]]=w
    checked.append(dict(上游站=s,下游站=r,路径河段数=len(chain),距离独立复算差_km=discrepancy,最大河段衔接差_m=endpoint_error,河段路径='>'.join(map(str,chain))))
rebuilt/=rebuilt.sum(1,keepdims=True)
np.testing.assert_allclose(rebuilt,a,atol=1e-12)
conservative=np.load(HERE/'河网拓扑_保守.npz')
np.testing.assert_array_equal(conservative['adjacency'],np.eye(750))
np.testing.assert_array_equal(ids,np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids'])
spec=importlib.util.spec_from_file_location('gwn',ROOT/'GraphWaveNet拓扑对照_20261109/source/model.py')
gwn=importlib.util.module_from_spec(spec);spec.loader.exec_module(gwn)
toy=torch.zeros(1,1,3,1);toy[0,0,0,0]=7
source_receiver=torch.zeros(3,3);source_receiver[0,2]=1
actual=gwn.nconv()(toy,source_receiver)
assert actual[0,0,2,0]==7 and actual[0,0,0,0]==0
pd.DataFrame(checked).to_csv(HERE/'逐连接独立核验.csv',index=False,encoding='utf-8-sig')
audit=dict(passed=True,checked_edges=len(checked),path_ids_traced=True,distance_checked_using_independent_outlet_distance=True,outlet_distance_tolerance='max(0.2 km, 1% of routed distance)',max_outlet_distance_discrepancy_km=max(r['距离独立复算差_km'] for r in checked),every_route_segment_junction_checked=True,author_source_to_receiver_direction_verified=True,station_order_verified=True,conservative_is_identity=True,station_locations_still_unverified=True)
(HERE/'河网核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
