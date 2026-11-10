"""Paired evaluation on the same stations, targets and timestamps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from graph import HERE, CACHE, CHEM

NAMES={"local":"单站LoRA","spatial":"跨站LoRA","ablated":"跨站权重_预测时移除邻站","persistence":"持久性"}

def save(frame,name):
    frame.to_csv(HERE/name,index=False,encoding="utf-8-sig",float_format="%.12g")

def table(frame):
    lines=["| "+" | ".join(frame.columns)+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    for row in frame.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(f"{v:.5f}" if isinstance(v,float) else str(v) for v in row)+" |")
    return "\n".join(lines)

def main():
    audit=json.loads((HERE/"训练核验.json").read_text(encoding="utf-8"))
    graph_audit=json.loads((HERE/"邻站输入独立核验.json").read_text(encoding="utf-8"))
    assert audit["passed"] and all(x["reconstructed_all_messages"] for x in graph_audit.values())
    z=np.load(HERE/"predictions.npz")
    idx=pd.read_csv(CACHE/"配对索引.csv").iloc[z["rows"]].reset_index(drop=True)
    assert idx.test.all() and len(idx)==23136 and idx["站点ID"].nunique()==750
    truth=z["truth"]
    rows=[]
    for key,name in NAMES.items():
        p=np.maximum(z[key],0)
        for sid,ix in idx.groupby("站点ID").indices.items():
            y=truth[ix];e=p[ix]-y;sst=np.square(y-y.mean(axis=0)).sum(axis=0)
            tol=1e-12*np.maximum(1,np.square(y).sum(axis=0))
            r2=1-np.divide(np.square(e).sum(axis=0),sst,out=np.full(4,np.nan),where=sst>tol)
            meta=idx.iloc[ix[0]]
            for j,chem in enumerate(CHEM):
                rows.append([name,sid,meta["流域"],meta["省份"],meta["断面"],chem,len(ix),
                             np.abs(e[:,j]).mean(),np.sqrt(np.square(e[:,j]).mean()),r2[j]])
    M=pd.DataFrame(rows,columns=["模型","站点ID","流域","省份","断面","指标","配对数","MAE","RMSE","R2"])
    prior=pd.read_csv(HERE.parent/"Chronos2_LoRA水质24小时评估_20261109/01_逐站点误差.csv")
    paired=M[M["模型"].eq("单站LoRA")].merge(prior[prior["模型"].eq("官方LoRA")],on=["指标","站点ID"],validate="one_to_one")
    for metric in ("R2","MAE","RMSE"):
        np.testing.assert_allclose(paired[metric+"_x"],paired[metric+"_y"],atol=1e-8,rtol=1e-8,equal_nan=True)
    save(M,"01_逐站点误差.csv")
    summary=M.groupby(["模型","指标"],sort=False).agg(有效站点=("R2","count"),平均R2=("R2","mean"),R2中位数=("R2","median"),MAE中位数=("MAE","median"),RMSE中位数=("RMSE","median"),超过05站=("R2",lambda x:x.gt(.5).sum()),超过08站=("R2",lambda x:x.gt(.8).sum())).reset_index()
    save(summary,"02_分指标结果.csv")
    totals=[]
    for name,g in M.groupby("模型",sort=False):
        totals.append([name,summary.loc[summary["模型"].eq(name),"平均R2"].mean(),g.loc[g.R2.gt(.5),"站点ID"].nunique(),g.loc[g.R2.gt(.8),"站点ID"].nunique()])
    totals=pd.DataFrame(totals,columns=["模型","四指标等权平均R2","任一指标超过0.5站数","任一指标超过0.8站数"])
    save(totals,"03_总体比较.csv")
    changes=[]
    detail=M[M["模型"].eq("跨站LoRA")]
    for name in ["单站LoRA","跨站权重_预测时移除邻站"]:
        paired=detail.merge(M[M["模型"].eq(name)],on=["站点ID","指标"],suffixes=("_跨站","_对照"),validate="one_to_one")
        paired["R2变化"]=paired.R2_跨站-paired.R2_对照
        paired["MAE降低量"]=paired.MAE_对照-paired.MAE_跨站
        changes.append(paired)
    changes=pd.concat(changes,ignore_index=True)
    save(changes,"04_逐站点邻站收益.csv")
    stats=changes.groupby(["模型_对照","指标"],sort=False).agg(R2可比较站数=("R2变化","count"),R2提高站数=("R2变化",lambda x:x.gt(1e-6).sum()),R2变化中位数=("R2变化","median"),MAE降低站数=("MAE降低量",lambda x:x.gt(1e-8).sum())).reset_index()
    save(stats,"05_逐指标改善站点.csv")
    basins=changes.groupby(["模型_对照","流域_跨站","指标"],sort=False).agg(站点数=("站点ID","nunique"),R2可比较站数=("R2变化","count"),R2变化中位数=("R2变化","median"),MAE降低量中位数=("MAE降低量","median")).reset_index()
    save(basins,"06_分流域邻站收益.csv")
    cover=json.loads((HERE/"图覆盖.json").read_text(encoding="utf-8"))["final"]
    numbers=totals.set_index("模型")
    single=numbers.loc["单站LoRA"];multi=numbers.loc["跨站LoRA"];masked=numbers.loc["跨站权重_预测时移除邻站"]
    report=f"""# 同流域跨站信息对24小时水质预测的影响

增加邻站历史后，四指标等权平均R²从{single["四指标等权平均R2"]:.5f}变为{multi["四指标等权平均R2"]:.5f}，变化{multi["四指标等权平均R2"]-single["四指标等权平均R2"]:+.5f}。至少一项R²>0.5的站点从{int(single["任一指标超过0.5站数"])}站变为{int(multi["任一指标超过0.5站数"])}站。使用同一跨站权重、预测时移除邻站输入，平均R²为{masked["四指标等权平均R2"]:.5f}。

参考本地PSIAL的model.py中RelationAwareSBDProjectionExchangeInteraction跨节点交换信息的设计，以及psial_ablation.py关闭空间交互的对照。水质实现沿用原Chronos-2 LoRA模型、优化器、数据窗口和评估口径。PSIAL本身通过学习注意力聚合站点信息。本次试验选用更便于检查的规则，先依据同期观测构建固定相关图，再让水质模型学习使用邻站信号。

## 建图和输入

每一阶段按训练时间截止值分开建图和标准化。候选为同一流域的其他站点。对每项化学指标，使用同一时刻的实测值计算Pearson相关系数，要求至少16对且r≥0.5。按r×n/(n+16)排序，每站每项最多保留3个邻站。正相关阈值、最少重叠数及邻站数是本次预设参数。统计关系描述同步变化，河网距离、上下游方向和传播时滞未进入模型。

在本站最近6条历史覆盖的时间网格上，用同一时刻有观测的邻站计算标准化加权平均，形成4条历史协变量。没有邻站观测的格子保留缺失。每条预测只访问起报时间及以前的邻站化学值；未来仅开放本站+24小时5项易测值。最终图有{cover["edges"]}条站点–指标关联，覆盖{cover["connected_stations"]}站。测试中{cover["test_pairs_with_history"]}/23,136个配对有邻站历史，{cover["test_pairs_with_origin"]}个在起报时刻有邻站观测。邻站输入按原始时间和边列表独立重算通过。

## 训练和对照

750站、79,940条源观测，使用27,130个选择训练窗口、9,107个验证窗口、38,171个重新拟合窗口。选模型时仅使用4月24日前的数据建图，最终图使用4月29日前的数据。与原单站LoRA同样的原始权重、随机种子、官方默认LoRA目标层、每批32个窗口、学习率1e-5和500步上限。空间模型验证选中第{audit["best_steps"]}步，重新加载适配器评估完全相同的23,136个测试配对。

单站LoRA取自已有实验，已经独立匹配全部3,000个站点指标误差。另一组对照用相同跨站权重，在预测时去掉4条邻站协变量。前者比较两种训练方案，后者检验模型对邻站输入的依赖。所有方案均使用相同非负后处理。官方损失按批中变量条数取平均，跨站方案每窗口13条、单站方案9条，模型间的比较以相同测试配对的原单位误差和R²为准。

## 结果

{table(totals)}

{table(summary)}

{table(stats)}

平均R²先在每项指标内逐站算术平均，再四项等权平均。阈值按750站中任一指标严格超过阈值去重；有限负R²全部保留。改善站点统计分别用1e-6的R²和1e-8的MAE容差。分流域结果在06_分流域邻站收益.csv。当前为同一30天数据的回溯评估，测试时间段此前已用于多轮比较。
"""
    (HERE/"实验报告.md").write_text(report,encoding="utf-8")
    print(totals.to_string(index=False));print(summary.to_string(index=False));print(stats.to_string(index=False))

if __name__=="__main__":
    main()
