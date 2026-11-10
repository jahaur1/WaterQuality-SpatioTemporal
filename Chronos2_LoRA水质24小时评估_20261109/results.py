from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from safetensors.numpy import load_file

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "冻结预训练模型校正头基线_20261106"
CHEM = ["高锰酸盐指数(mg/L)", "氨氮(mg/L)", "总磷(mg/L)", "总氮(mg/L)"]

def metrics(y, p):
    e = y-p
    sst = np.sum((y-y.mean())**2)
    r2 = 1 - np.dot(e,e)/sst if sst > 1e-12*max(1,np.dot(y,y)) else np.nan
    return np.abs(e).mean(), np.sqrt(np.mean(e**2)), r2

def table(df):
    lines = ["| "+" | ".join(df.columns)+" |", "| "+" | ".join(["---"]*len(df.columns))+" |"]
    for row in df.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(f"{v:.5f}" if isinstance(v,float) else str(v) for v in row)+" |")
    return "\n".join(lines)

def main():
    audit = json.loads((HERE/"核验.json").read_text(encoding="utf-8"))
    assert audit["passed"]
    tensors = load_file(str(HERE/"final/finetuned-ckpt/adapter_model.safetensors"))
    assert all("lora_" in k and np.isfinite(v).all() for k,v in tensors.items())
    assert sum(v.size for v in tensors.values())==audit["trainable"]
    assert any(np.any(v) for k,v in tensors.items() if "lora_B" in k)
    saved = np.load(HERE/"predictions.npz")
    idx = pd.read_csv(CACHE/"配对索引.csv").iloc[saved["rows"]].reset_index(drop=True)
    assert idx.test.all() and idx["站点ID"].nunique()==750
    observed = saved["truth"]
    persistence = np.load(CACHE/"inputs.npz")["last_chemical"][saved["rows"]]
    rows = []
    for label, pred in [("冻结原模型",saved["frozen"]),("官方LoRA",saved["raw"]),("持久性",persistence)]:
        pred = np.maximum(pred,0)
        for station, ids in idx.groupby("站点ID").indices.items():
            meta = idx.iloc[ids[0]]
            for j, name in enumerate(CHEM):
                mae, rmse, r2 = metrics(observed[ids,j],pred[ids,j])
                rows.append([label,station,meta["省份"],meta["流域"],meta["断面"],name,len(ids),mae,rmse,r2])
    m = pd.DataFrame(rows,columns=["模型","站点ID","省份","流域","断面","指标","配对数","MAE","RMSE","R2"])
    old = pd.read_csv(CACHE/"01_逐站点模型误差.csv")
    check = m[m["模型"].eq("冻结原模型")].merge(old[old["模型"].eq("Chronos2_冻结原模型")],left_on=["站点ID","指标"],right_on=["站点ID","目标变量"],validate="one_to_one")
    assert len(check)==3000
    for k in ["MAE","RMSE","R2"]:
        np.testing.assert_allclose(check[k+"_x"],check[k+"_y"],rtol=1e-8,atol=1e-8,equal_nan=True)
    m.to_csv(HERE/"01_逐站点误差.csv",index=False,encoding="utf-8-sig")
    summary = m.groupby(["模型","指标"],sort=False).agg(有效站点=("R2","count"),平均R2=("R2","mean"),R2中位数=("R2","median"),MAE中位数=("MAE","median"),RMSE中位数=("RMSE","median"),超过05站=("R2",lambda x:(x>.5).sum()),超过08站=("R2",lambda x:(x>.8).sum())).reset_index()
    summary.to_csv(HERE/"02_分指标结果.csv",index=False,encoding="utf-8-sig")
    totals = []
    for name,g in m.groupby("模型",sort=False):
        totals.append([name,summary[summary["模型"].eq(name)].平均R2.mean(),g.loc[g.R2.gt(.5),"站点ID"].nunique(),g.loc[g.R2.gt(.8),"站点ID"].nunique()])
    totals = pd.DataFrame(totals,columns=["模型","四指标等权平均R2","任一项R2大于0.5站数","任一项R2大于0.8站数"])
    totals.to_csv(HERE/"03_总体比较.csv",index=False,encoding="utf-8-sig")
    gains = m[m["模型"].eq("官方LoRA")].merge(m[m["模型"].eq("冻结原模型")],on=["站点ID","指标"],suffixes=("_LoRA","_冻结"),validate="one_to_one")
    gains["R2变化"]=gains.R2_LoRA-gains.R2_冻结
    gains["MAE降低量"]=gains.MAE_冻结-gains.MAE_LoRA
    gains.to_csv(HERE/"04_逐站点改善.csv",index=False,encoding="utf-8-sig")
    improved=gains.groupby("指标",sort=False).agg(R2可比较站数=("R2变化","count"),R2提高站数=("R2变化",lambda x:(x>0).sum()),MAE降低站数=("MAE降低量",lambda x:(x>0).sum())).reset_index()
    change=[]
    for cutoff in [.5,.8]:
        prior=set(gains.loc[gains.R2_冻结.gt(cutoff),"站点ID"])
        tuned=set(gains.loc[gains.R2_LoRA.gt(cutoff),"站点ID"])
        change.append({"阈值":cutoff,"新增站点":len(tuned-prior),"跌出站点":len(prior-tuned),"两者均达标":len(prior&tuned)})
    transition=pd.DataFrame(change)
    log=json.loads((HERE/"selection/training.json").read_text(encoding="utf-8"))
    val=pd.DataFrame([{"步数":r["step"],"验证分位数损失":r["eval_loss"]} for r in log["history"] if "eval_loss" in r])
    report=f"""# Chronos-2 官方 LoRA 水质评估

750站，23,136个精确24小时测试配对。输入为起报时最近6条9项观测构成的既有32格历史，仅第6个未来格可见5项易测真值。每站历史化学按实测更新。所有站点共享一个适配器。

直接调用本地 Chronos2Pipeline.fit 的 finetune_mode=lora，使用作者默认 r=8、alpha=16 和目标层，学习率1e-5。每次更新288个变量序列，即32个多变量窗口。每阶段最多500步，随机种子20261109，使用全精度。实际训练 {audit['trainable']:,} 个 LoRA 参数。化学训练标签只在第6格出现，其他5格缺测，由官方分位数损失掩码处理。

选择训练27,130对，标签早于4月24日。每100步用9,107个验证配对评估官方损失，选中第 {audit['selection_steps']} 步。从原权重重新初始化，使用4月29日前38,171对重新训练至该步数。测试起报自4月29日开始，目标时刻为4月30日至5月10日00:00。选择和重拟合保持相同的500步线性学习率计划。

{table(val)}

## 总体比较

{table(totals)}

先在各指标的可计算站点上平均R²，再对四指标等权平均。高R²按任一指标达标的站点去重。评分前统一将负浓度截为0，保留全部有限负R²。

## 分指标比较

{table(summary)}

## 相对冻结模型的配对改善

{table(improved)}

{table(transition)}

## 核验

原始权重SHA256保持一致，训练回调逐项确认所有可训练参数属于LoRA。已重新加载最终适配器做推断并重放首批预测。冻结原模型的3,000组误差与既有实验一致。测试时段此前已用于多轮探索，结果反映这750个已知站点在当前时段的表现。检验其他月份的表现需另取时间段。
"""
    (HERE/"实验报告.md").write_text(report,encoding="utf-8")
    print(totals.to_string(index=False));print(summary.to_string(index=False));print(improved.to_string(index=False));print(val.to_string(index=False));print(transition.to_string(index=False))

if __name__=="__main__":
    main()
