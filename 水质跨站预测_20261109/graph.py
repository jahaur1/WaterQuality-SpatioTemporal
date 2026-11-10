"""Build basin-specific station graphs using synchronous pre-cut measurements."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "冻结预训练模型校正头基线_20261106"
DATA = ROOT / "30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv"
CHEM = ["高锰酸盐指数(mg/L)", "氨氮(mg/L)", "总磷(mg/L)", "总氮(mg/L)"]
sys.stdout.reconfigure(encoding="utf-8")

def build(raw, idx, history, stage, cut):
    stations = pd.Index(sorted(raw["站点ID"].unique()))
    metadata = raw.drop_duplicates("站点ID").set_index("站点ID").loc[stations]
    times = pd.DatetimeIndex(sorted(raw["监测时间"].unique()))
    before = times < cut
    station_rows = stations.get_indexer(idx["站点ID"])
    hours = pd.to_datetime(idx["起报时间"], format="mixed").to_numpy()[:, None] - np.arange(31,-1,-1)*np.timedelta64(4,"h")
    grid_positions = times.get_indexer(hours.ravel()).reshape(hours.shape)
    allowed = (grid_positions>=0) & np.isfinite(history[:,:,0])
    spatial = np.full((*hours.shape,4), np.nan, dtype=np.float32)
    edges = []
    for j, target in enumerate(CHEM):
        frame = raw.pivot(index="监测时间",columns="站点ID",values=target).reindex(index=times,columns=stations)
        train = frame.loc[before]
        senders, receivers, scores = [], [], []
        for basin, group in metadata.groupby("流域"):
            ids = group.index
            block = train[ids]
            corr = block.corr(min_periods=16).to_numpy()
            observed = block.notna().to_numpy(dtype=np.int32)
            overlap = observed.T @ observed
            score = corr * overlap / (overlap+16.)
            eligible = np.isfinite(corr) & (corr>=.5) & (overlap>=16)
            np.fill_diagonal(eligible, False)
            for i, receiver in enumerate(ids):
                candidates = np.flatnonzero(eligible[i])
                ranked = sorted(candidates, key=lambda k:(-score[i,k],ids[k]))[:3]
                for k in ranked:
                    senders.append(stations.get_loc(ids[k]))
                    receivers.append(stations.get_loc(receiver))
                    scores.append(score[i,k])
                    edges.append([stage,basin,target,receiver,ids[k],overlap[i,k],corr[i,k],score[i,k]])
        weights = csr_matrix((scores,(receivers,senders)),shape=(len(stations),len(stations)))
        scales = train.std(ddof=0).replace(0,np.nan)
        values = ((frame-train.mean())/scales).to_numpy()
        valid = np.isfinite(values)
        numerator = (weights @ np.where(valid,values,0.).T).T
        denominator = (weights @ valid.astype(float).T).T
        message = np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        aligned = message[grid_positions.clip(min=0),station_rows[:,None]]
        spatial[:,:,j] = np.where(allowed,aligned,np.nan)
    columns=["阶段","流域","化学指标","接收站","邻站","同步观测数","Pearson_r","权重"]
    edge_frame=pd.DataFrame(edges,columns=columns)
    for role in ["接收站","邻站"]:
        edge_frame[role+"省份"]=edge_frame[role].map(metadata["省份"])
        edge_frame[role+"断面"]=edge_frame[role].map(metadata["断面名称"])
    assert not edge_frame["接收站"].eq(edge_frame["邻站"]).any()
    assert not np.isfinite(spatial[~allowed]).any()
    test=idx.test.to_numpy(bool)
    summary=dict(cut=cut,latest_graph_observation=str(times[before].max()),edges=len(edges),
                 connected_stations=edge_frame["接收站"].nunique(),
                 test_pairs_with_history=int(np.isfinite(spatial[test]).any(axis=(1,2)).sum()),
                 test_pairs_with_origin=int(np.isfinite(spatial[test,-1]).any(axis=1).sum()))
    edge_frame.to_csv(HERE/f"{stage}_关联站点.csv",index=False,encoding="utf-8-sig")
    np.save(HERE/f"{stage}_邻站历史.npy",spatial)
    print(stage,summary,flush=True)
    return summary

if __name__=="__main__":
    raw=pd.read_csv(DATA,parse_dates=["监测时间"])
    idx=pd.read_csv(CACHE/"配对索引.csv")
    hist=np.load(CACHE/"inputs.npz")["history"]
    assert len(raw)==79940 and raw["站点ID"].nunique()==750
    summaries={key:build(raw,idx,hist,key,cut) for key,cut in [("selection","2025-04-24"),("final","2025-04-29")]}
    (HERE/"图覆盖.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
