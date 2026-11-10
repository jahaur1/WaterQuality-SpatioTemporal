import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def table(df):
    out=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(out)

def main():
    info=json.loads((HERE/'距离建图记录.json').read_text(encoding='utf-8'));locations=pd.read_csv(HERE/'站点地图定位.csv').set_index('站点ID');assert len(locations)==750
    z=np.load(HERE/'predictions.npz');p=np.load(ROOT/'GraphWaveNet拓扑对照_20261109/predictions.npz');np.testing.assert_array_equal(z['truth'],p['truth'])
    idx=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv');idx=idx[idx.test].reset_index(drop=True)
    graph=np.load(HERE/'距离拓扑.npz');a=graph['adjacency'];nodes=graph['station_ids'];eligible=graph['eligible'];np.testing.assert_allclose(a.sum(1),1,atol=1e-10)
    assert np.all(np.count_nonzero(a,axis=1)[~eligible]==1)
    e=pd.read_csv(HERE/'距离关联站点.csv')
    assert (e['估算地理距离_km']<=250+1e-7).all();assert np.all(locations.loc[e['接收站'],'流域'].to_numpy()==locations.loc[e['邻站'],'流域'].to_numpy())
    positions={sid:i for i,sid in enumerate(nodes)}
    for row in e.sample(min(len(e),30),random_state=9).to_dict('records'):
        i=positions[row['接收站']];j=positions[row['邻站']]
        lon1,lat1=locations.loc[row['接收站'],['经度','纬度']];lon2,lat2=locations.loc[row['邻站'],['经度','纬度']]
        lon1,lat1,lon2,lat2=np.radians([lon1,lat1,lon2,lat2]);cos=np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)
        distance=6371*np.arccos(np.clip(cos,-1,1));np.testing.assert_allclose(distance,graph['distances_km'][i,j],atol=.001)
    values={'Pearson固定图':p['GWN固定图'],'距离固定图':z['GWN固定图'],'Pearson加自适应':p['GWN固定加自适应'],'距离加自适应':z['GWN固定加自适应']}
    rows=[]
    eligible_ids=set(nodes[eligible])
    for name,raw in values.items():
        pred=np.maximum(raw,0)
        for sid,ix in idx.groupby('站点ID').indices.items():
            y=z['truth'][ix];error=pred[ix]-y;sst=np.square(y-y.mean(0)).sum(0);tol=1e-12*np.maximum(1,np.square(y).sum(0));r2=1-np.divide(np.square(error).sum(0),sst,out=np.full(4,np.nan),where=sst>tol)
            for j,c in enumerate(CHEM):rows.append([name,sid,c,sid in eligible_ids,np.abs(error[:,j]).mean(),np.sqrt(np.square(error[:,j]).mean()),r2[j]])
    m=pd.DataFrame(rows,columns=['模型','站点ID','指标','有可用估算位置','MAE','RMSE','R2']);m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
    s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median')).reset_index();s.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
    totals=[]
    for scope,frame in [('全部750站',m),('有可用估算位置',m[m['有可用估算位置']])]:
        for name,g in frame.groupby('模型',sort=False):totals.append([scope,name,g['站点ID'].nunique(),g.groupby('指标').R2.mean().mean(),g.loc[g.R2.gt(.5),'站点ID'].nunique(),g.loc[g.R2.gt(.8),'站点ID'].nunique()])
    t=pd.DataFrame(totals,columns=['范围','模型','站点数','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数']);t.to_csv(HERE/'总体与定位子集比较.csv',index=False,encoding='utf-8-sig')
    summary=json.loads((HERE/'地图定位汇总.json').read_text(encoding='utf-8'))
    report=f'''# 公开地图粗定位与距离拓扑试验

## 位置资料

对现有750站按“省份+断面名称+水质监测站”查询百度公开地图，未取得地物候选时再按“省份+断面名称”查询。保留原始查询URL、候选名称、地址与行政区。优先采用同名监测设施，其次同名桥闸、村镇、水库等地物。桥/大桥等文字变体按相同地名匹配，同省多地同名且相距超过20km的候选单独标记。商业店铺和充电站等不用于桥闸定位。
所有点位是地图匹配的估算位置，均标记待人工核实。监测设施候选经验半径0.5km，同名桥闸2km，桥闸地名扩展和村镇10km，短地名或水库等20km。县区中心50km、市级中心100km，不纳入本轮距离图。经验半径是预设质量分级，未经过误差校准。不得据此将点位当作正式监测取水点。

地图搜索返回百度墨卡托坐标，用bd09convertor0.0.3统一转成BD09LL经纬度。来源https://map.baidu.com/ ，坐标转换工具https://pypi.org/project/bd09convertor/ 。本次地点查询日期2026-11-09，原水质时段为2025年4月至5月。站点地图定位.csv及map_cache保存可复核的检索记录。

{json.dumps(summary,ensure_ascii=False,indent=2)}

## 距离图

在有无明显同名歧义、经验半径不超过20km的站点之间，以Haversine公式估算球面距离。候选限同流域，最多连接最近3站，最大距离250km。连接权重exp(-(d/100km)^2)，加入自身权重1并按接收站归一化。位置不足或没有近邻的站点只保留自身连接。此为地理距离图，河道沿程距离和上下游方向没有纳入。坐标近似误差会影响近邻排序，距离关联站点.csv给出两端经验误差半径和距离是否落入该范围。

{json.dumps(info,ensure_ascii=False,indent=2)}

## 模型对照

继续使用Graph WaveNet作者图卷积与Chronos-2末两层LoRA，分别训练距离固定图和距离加自适应图，固定24次全图更新，沿用相同种子、训练起报采样、学习率和23,136个测试配对。Pearson两组取自上一轮同配置。距离固定图的连接覆盖率和节点度与Pearson图不同，结果同时受定位覆盖率、图稀疏度和距离选边影响。自适应矩阵沿用作者SVD初始化，允许全域学习连接。

{table(t)}

{table(s)}

全750站都参与训练和评分，缺少可用位置的站点没有被删除。定位子集结果单独列出，便于评估地图覆盖限制。平均R²先对每项指标的可计算站点平均，再四项等权平均，有限负值保留。

已核对同流域约束、250km截止、无位置节点自身连接、邻接矩阵行和，以及30条边的另一种球面距离计算结果。测试真值与上一轮逐值一致。本轮是基于粗估位置的探索性试验，站点地名歧义和估算误差仍需后续核实。
'''
    (HERE/'实验报告.md').write_text(report,encoding='utf-8')
    (HERE/'核验.json').write_text(json.dumps(dict(passed=True,test_truth_identical=True,all_750_retained=True,missing_locations_self_only=True,distance_edges_verified=30),indent=2),encoding='utf-8')
    print(t.to_string(index=False));print(info)

if __name__=='__main__':main()
