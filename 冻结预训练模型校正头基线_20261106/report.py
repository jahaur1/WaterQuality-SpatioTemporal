from pathlib import Path
import sys,json
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
def table(df):
    cols=df.columns.tolist();out=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(out)
def main():
    verify=json.loads((HERE/'独立核验.json').read_text(encoding='utf-8'));assert verify['passed']
    summary=pd.read_csv(HERE/'02_逐指标汇总.csv');overall=pd.read_csv(HERE/'06_总体比较.csv');cfg=json.loads((HERE/'训练配置与头权重.json').read_text(encoding='utf-8'))
    gains=pd.read_csv(HERE/'04_校正相对原模型逐站点收益.csv');gainrows=[]
    for (model,target),g in gains.groupby(['模型','目标变量'],sort=False):
        valid=g.R2提高量.notna();gainrows.append({'模型':model,'指标':target,'可比较R2站点数':int(valid.sum()),'R2提高站点数':int(g.R2提高量.gt(0).sum()),'R2提高量中位数':g.R2提高量.median(),'MAE降低站点数':int(g.MAE降低量.gt(0).sum())})
    gg=pd.DataFrame(gainrows);gg.to_csv(HERE/'07_逐指标改善站点统计.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    text='''# 冻结Timestral-209M与Chronos-2：共享线性残差头基线

本轮基线已执行并核验，整体没有改善：两个骨干的A/B校正头均使四指标等权平均R²更低，任一项R²>0.5和>0.8的站点数均少于各自冻结原模型；各项MAE中位数也均变差。部分站点或指标仍有R²提高，不等同于整体方案有效。尤其不能只用个别R²中位数变好掩盖MAE和大负R²恶化。

## 相关论文与本轮关系

1. Liang等，The Forecast After the Forecast: A Post-Processing Shift in Time Series（δ-Adapter，arXiv:2601.20280，论文页面注明ICLR 2026）。冻结原预测器，训练小型输出残差修正模块，论文还包含有界输入修正、特征选择和概率校准。本轮只参考输出残差校正思想，采用闭式岭回归，不声称完整复现其有界模块或训练细节。原文：https://arxiv.org/abs/2601.20280；全文：https://arxiv.org/html/2601.20280v1。
2. Im与Kwon，COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting（官方代码标注ICLR 2026）。通过原预测与近期真实观测统计生成线性残差，并用可学习门控制校正。其重点是在线测试时自适应；本轮是测试前离线拟合，不实现在线更新、门控或调度。官方说明及公式：https://github.com/bigbases/COSA_ICLR2026；论文：https://openreview.net/forum?id=L7Z5wBMPrW。

检索与实验日期：2026-11-06。上述相似性指冻结预测器后校正输出，不意味着论文在本水质数据上已有验证。

## 预先确定的基线结构

- 两个骨干均使用此前已下载的冻结权重和官方条件预测接口，不回传梯度、不微调骨干；采用输出的4项点预测，不提取隐藏表示。
- A，仅预测残差头：预测z + Linear(预测z_4→残差z_4)，20个可训练系数（16权重+4偏置）。
- B，结合观测残差头：预测z + Linear([预测z_4，起报化学z_4，目标时刻易测z_5]→残差z_4)，56个系数。
- 观测线性对照：起报化学z + Linear([起报化学z_4，目标时刻易测z_5]→变化z_4)，40个系数，不使用任何预训练预测。用于检验B的收益是否确实来自骨干输出。
- 同时保留两个冻结原模型和化学值持久性基线，合计8种输出。
- 每个头在750站之间共享，不使用站点ID、省份或流域作为输入；每站各自保存训练期9项均值与标准差，作为变换统计，不计入头网络参数。因此属于已有站点的跨时间评估，不是未见站点泛化。
- 采用带截距的残差岭回归。样本权重为1/该站训练配对数，使每站总权重相同；惩罚系数alpha=lambda×权重总和，截距不惩罚。lambda候选固定为[0.0001,0.001,0.01,0.1,1,10]，只按验证段逐站等权、四指标平均标准化MSE选择。
- 采用闭式解而不是迭代网络优化，因此没有轮数选择或随机初始化；零修正对应原模型，L2正则约束偏离原预测的幅度，但不施加δ-Adapter的硬信任域限制。

## 数据与时间控制

750站、79,940条原始观测。构造63,644个具有最近6条历史的精确+24小时配对；用于选择训练27,130对，验证9,107对，最终拟合38,171对，测试23,136对。
训练标签<2025-04-24；验证目标[2025-04-25,2025-04-29)，且起报>=4月24日；选择后用标签<4月29日重新拟合头；最终测试目标4月30日至5月10日00:00，起报>=4月29日。各阶段标准化仅拟合对应截止前的原始观测。
骨干历史仍为起报时最近6条9项真实观测：置于4小时网格，首条历史之前以NaN遮罩，6条历史首尾之间前向填充，32格上下文。预测未来6格，仅第6格（精确24小时）易测5项可见，中间5格易测值未知；未来化学真值不进入接口。每次预测允许使用截至起报最新化学实测值。
新增40,508个训练/验证等非测试窗口的冻结预测；原23,136个测试预测逐位复用，与此前比较完全一致。全部头拟合和lambda选择均不使用测试标签，测试期间不更新头。
相同的非负后处理用于原模型、校正头和观测对照；未截断输出也保留。

## 总体结果

四指标等权平均R²：先对每项可计算站点的R²求算术平均，再对四项等权平均。高R²站点按任一项达标去重，严格使用>，不是>=。不对负R²截尾，不把常量测试序列的无定义R²置零。有效R²站点数依次749（高锰酸盐）、623（氨氮）、737（总磷）、750（总氮）。

'''
    text+=table(overall)+'\n\n## 各项指标结果\n\n'+table(summary)+'\n\n## 相对各自原模型的改善\n\n'+table(gg)
    selected=pd.DataFrame([{'模型':r['model'],'可训练参数':r['trainable_parameters'],'lambda':r['selected_lambda']} for r in cfg['models']]);text+='\n\n## 验证选择\n\n'+table(selected)
    text+='''

## 极端误差核对

S1273的氨氮测试59对，实测均值约0.020169 mg/L、标准差约0.001291 mg/L。Timestral+B的RMSE约0.792353 mg/L，单站R²约-376801.667；Chronos2+B的RMSE约0.626794 mg/L，R²约-235789.242。原训练段此站氨氮近乎常量，StandardScaler按默认规则将尺度置1，未经幅度约束的跨站共享校正可产生不合适的较大改动。这既有预测误差增大，也有测试方差很小放大R²负值的因素，不能仅归因于R²统计特性。
08_氨氮极端误差核对.csv保存各方案最差氨氮站点的实测波动、预测范围与RMSE。所有极端值保留，未按测试表现剔除站点或截尾，也未在看到测试结果后改变正则化候选。此结果提示后续可另立实验研究有界修正、门控或针对恒定训练指标的尺度处理；本轮不追加这些方法来追逐已观察的测试结果。

## 核验与解释范围

独立从原数据重建全部63,644个窗口、目标值与条件输入；核验750站两阶段共1500份标准化统计；使用独立NumPy加权正规方程复算所有候选lambda的验证误差、最终系数和全部头测试预测；复算24,000组逐站点指标。两骨干权重哈希不变，原测试预测完全一致。详见独立核验.json。
本轮仅是线性输出校正基线，不是δ-Adapter或COSA完整复现，也不代表隐藏层探测或LoRA。所有方案同时报告，不依据测试集选一个作为事前确定的最优方案。
R²均值易受低波动站点的大负值影响，应结合中位数、MAE及高R²站点数阅读；改善站点比例只是数值改善，不是统计显著比例或推导成功率。
未来易测值是回溯实测条件，化学历史持续实测更新；本实验不能证明减少化学检测次数。骨干预训练数据与本地数据是否重合未核验。
同一测试时段此前已有多轮模型探索，本轮虽未以测试标签选头参数，仍应在新时段重复验证后形成确认性结论。

文件：01逐站点模型误差；02逐指标汇总；03高R²站点统计；04校正相对原模型逐站点收益；05分流域汇总；06总体比较；07逐指标改善站点统计。训练配置与头权重.json保存所有头系数、选择记录和输入哈希，test_predictions.npz保存未截断预测。
'''
    (HERE/'实验报告.md').write_text(text,encoding='utf-8');print(overall.to_string(index=False));print(gg.to_string(index=False))
if __name__=='__main__':main()
