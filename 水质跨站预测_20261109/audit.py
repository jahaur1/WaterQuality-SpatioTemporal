"""Independently rebuild historical messages from source rows and exported edges."""
import json
import numpy as np
import pandas as pd
from graph import HERE, DATA, CACHE, CHEM

def audit_graph(stage, cut, raw, idx, hist):
    edges=pd.read_csv(HERE/f"{stage}_关联站点.csv")
    meta=raw.drop_duplicates("站点ID").set_index("站点ID")["流域"]
    assert (edges["接收站"].map(meta)==edges["邻站"].map(meta)).all()
    assert (edges["接收站"]!=edges["邻站"]).all()
    groups={sid:g.set_index("监测时间").sort_index()[CHEM] for sid,g in raw.groupby("站点ID")}
    expected_times=pd.to_datetime(idx["起报时间"],format="mixed").to_numpy()[:,None]-np.arange(31,-1,-1)*np.timedelta64(4,"h")
    actual=np.load(HERE/f"{stage}_邻站历史.npy",mmap_mode="r")
    missing=np.full((len(idx),32),True)
    by_station=idx.groupby("站点ID").indices
    checked_links=0
    for j,chem in enumerate(CHEM):
        missing.fill(True)
        for sid,links in edges[edges["化学指标"].eq(chem)].groupby("接收站"):
            ix=by_station[sid]
            times=expected_times[ix].ravel()
            numerator=np.zeros(len(times));denominator=np.zeros(len(times))
            receiver=groups[sid].loc[groups[sid].index<cut,chem]
            assert len(links)<=3
            for row in links.to_dict("records"):
                series=groups[row["邻站"]][chem]
                train=series[series.index<cut]
                common=pd.concat([receiver,train],axis=1,join="inner").dropna()
                assert len(common)==row["同步观测数"] and len(common)>=16
                r=np.corrcoef(common.iloc[:,0],common.iloc[:,1])[0,1]
                np.testing.assert_allclose(r,row["Pearson_r"],atol=1e-8)
                assert r>=.5-1e-8
                weight=r*len(common)/(len(common)+16)
                values=series.reindex(times).to_numpy()
                present=np.isfinite(values)
                numerator[present]+=weight*(values[present]-train.mean())/train.std(ddof=0)
                denominator[present]+=weight
                checked_links+=1
            expected=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0).reshape(len(ix),32)
            expected[~np.isfinite(hist[ix,:,0])]=np.nan
            np.testing.assert_allclose(actual[ix,:,j],expected,rtol=1e-5,atol=1e-5,equal_nan=True)
            missing[ix]=False
        assert np.isnan(actual[:,:,j][missing]).all()
    return dict(edges=checked_links,all_windows=len(idx),channels=4,source_timestamps_no_later_than_origin=True,
                same_basin=True,self_edges=0,reconstructed_all_messages=True)

if __name__=="__main__":
    raw=pd.read_csv(DATA,parse_dates=["监测时间"]);idx=pd.read_csv(CACHE/"配对索引.csv")
    hist=np.load(CACHE/"inputs.npz")["history"]
    info={s:audit_graph(s,t,raw,idx,hist) for s,t in [("selection","2025-04-24"),("final","2025-04-29")]}
    (HERE/"邻站输入独立核验.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(info)
