from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def number(v,n=5):return f'{float(v):.{n}g}' if pd.notna(v) else 'NA'
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args();out=HERE/a.results
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));config=json.loads((out/'config.json').read_text(encoding='utf-8'));M=pd.read_csv(out/'01_逐站点两分支四目标误差.csv');coverage=pd.read_csv(out/'03_站点配对覆盖与不可评估原因.csv')
    f=M[M['分支']=='conditional_future_easy'];h=M[M['分支']=='history_only'];assert len(f)==len(h)==4*summary['基础站点数']
    merged=f.merge(h,on=['站点ID','目标变量'],suffixes=('_未来','_仅历史'),validate='one_to_one');rows=[]
    for r in merged.to_dict('records'):
        base=r['MAE_仅历史'];pers=r['起报时化学值持久性_MAE_未来']
        rows.append({'站点ID':r['站点ID'],'省份':r['省份_未来'],'流域':r['流域_未来'],'断面':r['断面_未来'],'目标变量':r['目标变量'],'测试配对数':r['精确24小时测试配对数_未来'],'未来易测真值_MAE':r['MAE_未来'],'无未来信息_MAE':base,'持久性_MAE':pers,'未来信息MAE绝对改善':base-r['MAE_未来'],'未来信息MAE相对改善':1-r['MAE_未来']/base if base>0 else np.nan,'有未来易测真值_R2':r['R2_未来'],'无未来信息_R2':r['R2_仅历史'],'持久性_R2':r['起报时化学值持久性_R2_未来'],'说明':'配对数值比较；不是可推导合格率'})
    R=pd.DataFrame(rows);R.to_csv(out/'07_逐站点未来信息增益对照.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    tables=[]
    for target in CHEM:
        g=R[R['目标变量']==target];mf=f[f['目标变量']==target]
        tables.append({'目标变量':target,'站点数':len(g),'条件MAE中位数':g['未来易测真值_MAE'].median(),'无未来MAE中位数':g['无未来信息_MAE'].median(),'持久性MAE中位数':g['持久性_MAE'].median(),'条件RMSE中位数':mf['RMSE'].median(),'条件R2中位数':g['有未来易测真值_R2'].median(),'无未来R2中位数':g['无未来信息_R2'].median(),'持久性R2中位数':g['持久性_R2'].median(),'未来信息使MAE数值降低站点数':int((g['未来信息MAE绝对改善']>0).sum()),'未来信息MAE相对改善中位数':g['未来信息MAE相对改善'].median(),'条件MAE低于持久性站点数':int((g['未来易测真值_MAE']<g['持久性_MAE']).sum())})
    T=pd.DataFrame(tables);T.to_csv(out/'08_核心比较汇总.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    source=json.loads((HERE/'source_provenance.json').read_text(encoding='utf-8'))
    report=f'''# XLinear：精确24小时条件预测评估

## 任务与数据

本次使用全部{summary['基础站点数']}个基础站点。每次预测在起点t的观测信息已获得后发起：输入为截至t最近6次的5项易测与4项化学实测历史，另给t+24小时的5项易测真值；输出t+24小时四化学指标。历史化学每次真实更新，不模拟减少化学检测。未来易测真值属于回溯条件信息，不是当时已拥有的预报。

严格只匹配同站真实存在的t与t+24小时，不插值或选最近时刻。满足6次历史条件的配对共{summary['6次历史可用精确配对数']:,}组，测试独立配对{summary['全部测试精确配对数']:,}组；两分支重复保存的行数为{summary['两分支测试记录合计']:,}，不是两倍独立测试样本。

可评估站点{summary['可评估站点数']}个、不可评估{summary['不可评估站点数']}个，基础名单均保留。测试配对最少{coverage['test_配对数'].min()}、中位{coverage['test_配对数'].median():g}、最多{coverage['test_配对数'].max()}，少于20组的站点{int((coverage['test_配对数']<20).sum())}个。

## 对照设定

- 条件模型：原生XLinear M模式9历史通道，保留历史窗口归一化，取四化学输出，加5→16→4未来易测分支。
- 无未来信息模型：同样历史9项和训练安排，关闭未来易测分支，不能使用目标时刻易测真值。
- 持久性基线：直接把每个起报时刻t的化学实测当作t+24小时预测；随起报时刻真实更新，不是前20天最后一次值一直保持。
- 训练期化学均值基线另存明细。

两模型各自只按验证损失选epoch，不在测试后挑选较好模型作为结果。未来分支末层零初始化，两条件同站使用相同初始种子。骨干原文件的字节级出处见source_provenance.json；这仍是明确添加未来分支及时间配对的任务适配，不是直接复现论文默认数据与超参数。

## 24小时预测的时间隔离

|阶段|化学标签时间|信息边界|
|---|---|---|
|选择epoch训练|严格早于4月24日|所有标准化统计仅原始时间<4月24日|
|验证|4月25日至4月29日之前|起报时间=标签时间−24小时，最早4月24日|
|重新初始化后重拟合|标签严格早于4月29日|所有标准化统计仅原始时间<4月29日|
|最终测试|4月30日至5月10日00:00含端点|起报时间4月29日至5月9日00:00|

训练标签以及用于选择epoch的验证标签都必须早于最早测试起报，不仅是早于测试目标。这避免模型在4月29日00:00发起预测时已看过4月29日稍晚的化学实测调参。

每站StandardScaler拟合9历史变量；未来5项用前5列训练均值和尺度，四化学输出用后4列训练均值和尺度。原生模型窗口均值/标准差只来自历史9项，不使用未来化学。测试化学真值可在之后的起报时刻进入历史，这是用户确认的滚动真实更新制度，不是参数重训练。

## 总体结果

各站单独评价，再取中位数；MAE与RMSE单位均为mg/L，R²无单位。三条件完全同站、同起报和目标时间、同标签。R²为常量/近常量测试目标时留空，保留负R²。

|目标|已知未来易测值MAE|无未来信息MAE|持久性MAE|已知未来易测值R²|
|---|---:|---:|---:|---:|
'''+''.join(f"|{r['目标变量']}|{number(r['条件MAE中位数'])}|{number(r['无未来MAE中位数'])}|{number(r['持久性MAE中位数'])}|{number(r['条件R2中位数'])}|\n" for r in tables)+'''
## 未来协变量带来的配对变化

|目标|未来信息使MAE数值降低的站点|站点相对MAE改善中位数|条件模型MAE低于持久性的站点|
|---|---:|---:|---:|
'''+''.join(f"|{r['目标变量']}|{r['未来信息使MAE数值降低站点数']}／{r['站点数']}|{r['未来信息MAE相对改善中位数']:.2%}|{r['条件MAE低于持久性站点数']}／{r['站点数']}|\n" for r in tables)+'''
上述只是误差数值比较，可能包含极小差异，不代表统计显著、业务达标或可减少检测比例。相对改善遇到无未来基线MAE=0时不计算，保留绝对差。

## 分流域结果

|流域|目标|条件MAE|无未来MAE|持久性MAE|条件R²|
|---|---|---:|---:|---:|---:|
'''
    for (basin,target),g in R.groupby(['流域','目标变量']):
        report+=f"|{basin}|{target}|{number(g['未来易测真值_MAE'].median())}|{number(g['无未来信息_MAE'].median())}|{number(g['持久性_MAE'].median())}|{number(g['有未来易测真值_R2'].median())}|\n"
    if all(r['条件MAE中位数']>r['无未来MAE中位数'] and r['条件MAE中位数']>r['持久性MAE中位数'] for r in tables):
        report+='\n## 本轮结论\n\n当前配置下，四项条件模型的站点MAE中位数均高于无未来信息模型及持久性基线，未观察到加入未来易测真值后的总体MAE改善。这不证明未来信息本身无效，也不证明其他架构、训练安排或更多独立数据下无法改善；不能根据本轮测试成绩再挑选窗口或配置而继续把同一测试集当作未见数据。\n'
    report+='''
## 解释边界与文件

- 未来易测真值在真实提前预测时通常尚未获得；本实验是已知未来信息的回溯条件场景，不是可部署提前预报能力，也不是数学最优性能上界。
- 每次使用化学真值更新历史意味着仍有持续化学实测，没有证明降低检测频率；未另模拟化学化验报告延迟。
- 对照只隔离在这套架构和训练流程中加入未来易测分支的影响；无未来分支模型拥有化学历史，不能与此前仅5项易测输入的同期软测量结果直接作同任务架构比较。
- 750站和短期不等距序列不是2500站、全年或新站迁移；样本此前已做过相关探索，窗口重叠和时间自相关使测试配对不独立。
- 未设定业务允许误差，任何“MAE较低站点比例”均不是可推导成功率；同一数据集也不用于据测试成绩继续挑模型、窗口或变量。
- R²按每站测试段单独计算，负值表示平方误差大于使用该站测试实测均值作常数的误差；不等同于比可部署的持久性基线更差。测试值变化很小时R²可能不稳定，不能单凭R²判定业务可用。

文件：01为两分支750站×4化学指标误差（共6000行），02为逐配对预测与实测，03为站点覆盖与不可评估理由，04和05为总体及分流域汇总，06为训练和时间记录，07为未来信息逐站配对增益，08为核心比较。stations目录保留两分支模型、损失记录和scaler，config/environment记录配置和环境。
'''
    audit=out/'独立验证.json'
    if audit.exists():
        data=json.loads(audit.read_text(encoding='utf-8'));assert data.get('passed') is True
        report+='\n## 独立复核\n\n独立验证通过。逐站核对精确24小时配对、历史与未来输入列、起报时间信息边界、训练期标准化和保存指标复算，并恢复抽样模型在CPU上复核预测。完整记录见独立验证.json。\n'
    (out/'24小时条件预测评估报告.md').write_text(report,encoding='utf-8')
    print(T.to_string(index=False))
if __name__=='__main__':main()
